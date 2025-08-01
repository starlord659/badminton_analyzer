# ==============================================================================
# coach_analyzer.py (Version 26 - Simplified & Robust Trigger)
#
# This version reverts to a single, universal motion-based trigger while keeping
# the advanced state machine, providing a "best of both worlds" solution
# for reliability and dynamic adaptation.
# ==============================================================================
import os
import json
import argparse
from collections import defaultdict, deque
import torch
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from train_posec3d import LitePoseC3D, create_pose_video_from_kpts, \
                              SEQUENCE_LENGTH, NUM_KEYPOINTS, HEATMAP_SIZE
except ImportError:
    # Fallback definitions...
    print("Warning: Could not import from 'train_posec3d.py'. Using fallback definitions.")
    SEQUENCE_LENGTH, NUM_KEYPOINTS, HEATMAP_SIZE = 32, 17, (64, 64)
    class LitePoseC3D(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return self.fc(torch.randn(1,1))
    def create_pose_video_from_kpts(*args, **kwargs): return np.zeros((NUM_KEYPOINTS, SEQUENCE_LENGTH, HEATMAP_SIZE[0], HEATMAP_SIZE[1]))

# --- CONSTANTS FOR TUNING (SIMPLIFIED) ---
VERBOSE_LOG = True
MOTION_THRESH = 37             # The single, universal threshold for shot detection.
PREPARE_MOTION_THRESH = 20      # Motion energy to enter the 'PREPARE' state.
RECOVERY_MOTION_THRESH = 20     # Motion energy to leave the 'RECOVERY' state.
IDLE_TIME_THRESH_SECONDS = 1.0
IDLE_MOTION_THRESH = 30

# Keypoint Constants
KP_L_SH, KP_R_SH, KP_L_EL, KP_R_EL, KP_L_WR, KP_R_WR, KP_L_AN, KP_R_AN = 5, 6, 7, 8, 9, 10, 15, 16
ACTION_KEYPOINT_INDICES = list(range(5, 17))

# --- HELPER FUNCTIONS ---
def draw_debug_panel(frame):
    panel_x, panel_y, panel_w, panel_h = 10, 10, 240, 90
    sub_img = frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
    frame[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = res
    cv2.putText(frame, "--- Tuning Constants ---", (panel_x + 5, panel_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Motion Thresh: {MOTION_THRESH}", (panel_x + 5, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Prepare Motion Thresh: {PREPARE_MOTION_THRESH}", (panel_x + 5, panel_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Recovery Motion Thresh: {RECOVERY_MOTION_THRESH}", (panel_x + 5, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

def select_roi(video_source):
    cap=cv2.VideoCapture(video_source); ret,frame=cap.read(); cap.release()
    if not ret: return None
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI"); return roi if roi[2]>0 and roi[3]>0 else None

def draw_static_gui(frame, gui_data):
    overlay = frame.copy()
    table_x, table_y, table_w, table_h = 10, 140, 220, 100
    cv2.rectangle(overlay, (table_x, table_y), (table_x + table_w, table_y + table_h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    player_ids = sorted(gui_data.keys())[:2]
    for i, track_id in enumerate(player_ids):
        data = gui_data[track_id]
        start_x = table_x + 10 + (i * 110)
        cv2.putText(frame, f"Player {track_id}", (start_x, table_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"- Motion: {data.get('motion', 0):>4.0f}", (start_x, table_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"- State: {data.get('state', 'N/A'):>7}", (start_x, table_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

def log_successful_shot(frame,state,val,thresh,shot):
    print(f"[SUCCESS] F:{frame}|P:{state.track_id}|Shot:{shot.upper()}|Val:{val:.1f}>Thresh:{thresh}")

class PlayerState:
    def __init__(self, track_id, fps):
        self.track_id=track_id; self.video_fps=fps
        self.keypoint_buffer=deque(maxlen=SEQUENCE_LENGTH)
        self.raw_motion_buffer=deque(maxlen=5)
        self.prev_kpts=None
        self.motion_energy=0.0
        self.status="Active"; self.last_shot="None"; self.last_shot_countdown=0
        self.idle_frame_counter=0; self.idle_time_thresh_frames=int(IDLE_TIME_THRESH_SECONDS*fps)
        self.fsm_state = "READY"

    def update_kinematics(self, kpts, frame_dims):
        frame_w, frame_h = frame_dims
        raw_motion = 0
        if self.prev_kpts is not None:
            c_kpts,p_kpts=kpts[ACTION_KEYPOINT_INDICES],self.prev_kpts[ACTION_KEYPOINT_INDICES]
            valid=((c_kpts[:,2]>0.1)&(p_kpts[:,2]>0.1)).cpu().numpy()
            if np.any(valid): raw_motion=np.sum(np.linalg.norm(c_kpts[:,:2].cpu().numpy()[valid]-p_kpts[:,:2].cpu().numpy()[valid], axis=1))
        
        self.raw_motion_buffer.append(raw_motion)
        self.motion_energy = np.mean(self.raw_motion_buffer)
        
        if self.motion_energy < IDLE_MOTION_THRESH: self.idle_frame_counter+=1
        else: self.idle_frame_counter=0
        self.status = "Idle" if self.idle_frame_counter > self.idle_time_thresh_frames else "Active"
        
        if self.status == "Active":
            kpts_norm=kpts[:,:2].cpu().numpy(); kpts_norm[:,0]/=frame_w; kpts_norm[:,1]/=frame_h
            self.keypoint_buffer.append(kpts_norm)
            
        self.prev_kpts = kpts.clone()
    
    def update_state_machine(self):
        if self.status == "Idle":
            self.fsm_state = "READY"
        
        if self.fsm_state == "READY":
            if self.motion_energy > PREPARE_MOTION_THRESH:
                self.fsm_state = "PREPARE"
        elif self.fsm_state == "PREPARE":
            is_triggered, trigger_val = self.check_trigger()
            if is_triggered:
                self.fsm_state = "RECOVERY"
                return True, trigger_val
        elif self.fsm_state == "RECOVERY":
            if self.motion_energy < RECOVERY_MOTION_THRESH:
                self.fsm_state = "READY"
        
        return False, 0

    def check_trigger(self):
        if len(self.raw_motion_buffer) < 3: return False, 0
        energies = list(self.raw_motion_buffer)
        is_peak = energies[-2] > 0.1 and energies[-2] >= max(energies)
        if not is_peak: return False, 0
        
        peak_energy = energies[-2]
        if peak_energy > MOTION_THRESH:
            return True, peak_energy
            
        return False, 0

    def set_last_shot(self,name): self.last_shot=name; self.last_shot_countdown=int(self.video_fps*2.5)
    def tick(self):
        if self.last_shot_countdown>0: self.last_shot_countdown-=1
        else: self.last_shot="None"

def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pose-weights',type=str,default='models/yolov8s-pose.pt')
    parser.add_argument('--classifier-weights',type=str,default='models/posec3d_classifier.pth')
    parser.add_argument('--class-map',type=str,default='models/class_map.json')
    parser.add_argument('--source',required=True,type=str)
    parser.add_argument('--device',type=str,default='cuda')
    return parser.parse_args()

def run_coach(opt):
    roi=select_roi(opt.source)
    device=torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    pose_model=YOLO(opt.pose_weights)
    with open(opt.class_map,'r') as f: class_to_idx=json.load(f)
    idx_to_class={i:name for name,i in class_to_idx.items()}
    classifier=LitePoseC3D(len(idx_to_class)).to(device)
    classifier.load_state_dict(torch.load(opt.classifier_weights,map_location=device)); classifier.eval()
    print("All models loaded. Starting analysis...")
    cap=cv2.VideoCapture(opt.source)
    fps,w,h=cap.get(cv2.CAP_PROP_FPS),int(cap.get(3)),int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    player_states={}; frame_num=0
    is_paused = True

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret: break
        annotated_frame = frame.copy()
        if roi: cv2.rectangle(annotated_frame, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (0,255,0), 1)
        for state in player_states.values(): state.tick()
        
        gui_data = defaultdict(dict)
        results=pose_model.track(frame,persist=True,verbose=False,device=device)
        active_ids=set()
        if results and results[0].boxes and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            kpts_data = results[0].keypoints.data
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()

            if len(track_ids) == kpts_data.shape[0]:
                kpts_data_map = {tid: data for tid, data in zip(track_ids, kpts_data)}
                boxes_map = {tid: data for tid, data in zip(track_ids, boxes_xyxy)}

                for track_id in track_ids:
                    active_ids.add(track_id)
                    box=boxes_map[track_id]
                    if roi and not (box[0]<roi[0]+roi[2] and box[2]>roi[0] and box[1]<roi[1]+roi[3] and box[3]>roi[1]): continue
                    if track_id not in player_states: player_states[track_id]=PlayerState(track_id,fps)
                    state=player_states[track_id]
                    state.update_kinematics(kpts_data_map[track_id], (w,h))
                    
                    gui_data[track_id].update({'motion': state.motion_energy, 'state': state.fsm_state})
                    is_triggered, trigger_val = state.update_state_machine()
                    
                    if VERBOSE_LOG:
                        log_str = (f"F:{frame_num}|P:{track_id}|"
                                   f"FSM:{state.fsm_state:<8}|S:{state.status:<6}|"
                                   f"M:{state.motion_energy:>4.1f}")
                        print(log_str)
                    
                    if is_triggered and len(state.keypoint_buffer)==SEQUENCE_LENGTH:
                        pose_video=create_pose_video_from_kpts(list(state.keypoint_buffer),SEQUENCE_LENGTH,NUM_KEYPOINTS,HEATMAP_SIZE)
                        input_tensor=torch.tensor(pose_video).unsqueeze(0).to(device)
                        with torch.no_grad(): shot_name=idx_to_class[torch.argmax(classifier(input_tensor)).item()]
                        state.set_last_shot(shot_name.replace('_',' '))
                        log_successful_shot(frame_num,state,trigger_val,MOTION_THRESH,shot_name)
                    
                    x1,y1=int(box[0]),int(box[1])
                    color=(0,255,0) if state.status=='Active' else (0,255,255)
                    cv2.rectangle(annotated_frame,(x1,y1),(int(box[2]),int(box[3])),color,1)
                    cv2.putText(annotated_frame,f"P{track_id} [{state.fsm_state}]",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                    if state.last_shot_countdown>0: cv2.putText(annotated_frame,state.last_shot,(x1,y1-25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        
        inactive_ids=set(player_states.keys())-active_ids
        for inactive_id in inactive_ids: del player_states[inactive_id]
        
        annotated_frame = draw_debug_panel(annotated_frame)
        annotated_frame = draw_static_gui(annotated_frame, gui_data)
        status_text = f"Frame: {frame_num}/{total_frames} | {'PAUSED' if is_paused else 'PLAYING'}"
        cv2.putText(annotated_frame, status_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Interactive AI Coach Debugger", annotated_frame)
        
        key = cv2.waitKey(1 if not is_paused else 0) & 0xFF
        if key == ord('q'): break
        elif key == ord(' '): is_paused = not is_paused
        
        if is_paused:
            if key == ord('.'): frame_num = min(frame_num + 1, total_frames - 1)
            elif key == ord(','): frame_num = max(frame_num - 1, 0)
        else:
            frame_num += 1
            
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_coach(parse_opt())