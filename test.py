# ==============================================================================
# coach_analyzer.py (Version 17.2 - "Stuck Zone" & "Dominant Arm" Fix)
#
# Description:
# This version fixes the "Z:N/A" bug that was preventing all triggers.
#
# Key Fixes:
# 1. SAFER DEFAULT ZONE: Players now default to the "BACK" zone instead of
#    "N/A", ensuring the back court trigger can always be evaluated.
# 2. ROBUST DOMINANT ARM DETECTION: The system now calculates the motion of
#    both arms and chooses the one that moved more to calculate Pose Velocity.
#    This makes the "V:" value in the log far more reliable.
# ==============================================================================
import os
import json
import argparse
from collections import deque
import torch
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from train_posec3d import LitePoseC3D, create_pose_video_from_kpts, \
                              SEQUENCE_LENGTH, NUM_KEYPOINTS, HEATMAP_SIZE
except ImportError:
    print("Warning: Could not import from 'train_posec3d.py'. Using fallback definitions.")
    SEQUENCE_LENGTH, NUM_KEYPOINTS, HEATMAP_SIZE = 32, 17, (64, 64)
    class LitePoseC3D(torch.nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return self.fc(torch.randn(1,1))
    def create_pose_video_from_kpts(*args, **kwargs): return np.zeros((NUM_KEYPOINTS, SEQUENCE_LENGTH, HEATMAP_SIZE[0], HEATMAP_SIZE[1]))

# --- CONSTANTS FOR TUNING ---
VERBOSE_LOG = False
BACK_COURT_MOTION_THRESH = 100
NET_ZONE_POSE_VELOCITY_THRESH = 20
MOTION_PEAK_WINDOW = 5
IDLE_MOTION_THRESH = 40 # Lowered this based on your previous logs
IDLE_TIME_THRESH_SECONDS = 1.0

# Keypoint Constants
KP_L_SH, KP_R_SH, KP_L_EL, KP_R_EL, KP_L_WR, KP_R_WR, KP_L_AN, KP_R_AN = 5, 6, 7, 8, 9, 10, 15, 16
ACTION_KEYPOINT_INDICES = list(range(5, 17))
LEFT_ARM_INDICES = [KP_L_SH, KP_L_EL, KP_L_WR]
RIGHT_ARM_INDICES = [KP_R_SH, KP_R_EL, KP_R_WR]

# Helper Functions
def select_roi(video_source):
    cap=cv2.VideoCapture(video_source); ret,frame=cap.read(); cap.release()
    if not ret: return None
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI"); return roi if roi[2]>0 and roi[3]>0 else None

def get_keypoint(kpts, i): return kpts[i] if kpts is not None and i < len(kpts) else (0,0,0)
def calculate_angle(p1,p2,p3):
    if p1[2]<0.1 or p2[2]<0.1 or p3[2]<0.1: return None
    v1,v2=np.array(p1[:2])-np.array(p2[:2]), np.array(p3[:2])-np.array(p2[:2])
    n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
    if n1==0 or n2==0: return None
    return np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)))
def get_court_zone(y,roi,h): return ("NET" if y<(roi[1]+roi[3]*0.5) else "BACK") if roi else ("NET" if y<h/2 else "BACK")
def log_successful_shot(frame,state,trigger,val,thresh,shot):
    print(f"[SUCCESS] F:{frame}|P:{state.track_id}|Shot:{shot.upper()}|Trigger:{trigger}|Val:{val:.1f}>Thresh:{thresh}|Zone:{state.zone}")

class PlayerState:
    def __init__(self, track_id, fps):
        self.track_id=track_id; self.video_fps=fps
        self.keypoint_buffer=deque(maxlen=SEQUENCE_LENGTH)
        self.motion_energy_buffer=deque(maxlen=MOTION_PEAK_WINDOW)
        self.prev_kpts=None; self.prev_elbow_angle=None
        self.cooldown_timer=0; self.motion_energy=0; self.pose_velocity=0
        self.zone="BACK" # --- FIX: Safer default ---
        self.status="Active"; self.last_shot="None"; self.last_shot_countdown=0
        self.idle_frame_counter=0; self.idle_time_thresh_frames=int(IDLE_TIME_THRESH_SECONDS*fps)

    def update_kinematics(self, kpts, kpts_unnorm, frame_dims, roi):
        frame_w, frame_h = frame_dims
        self.motion_energy = 0
        if self.prev_kpts is not None:
            c_kpts,p_kpts=kpts[ACTION_KEYPOINT_INDICES],self.prev_kpts[ACTION_KEYPOINT_INDICES]
            valid=((c_kpts[:,2]>0.1)&(p_kpts[:,2]>0.1)).cpu().numpy()
            if np.any(valid): self.motion_energy=np.sum(np.linalg.norm(c_kpts[:,:2].cpu().numpy()[valid]-p_kpts[:,:2].cpu().numpy()[valid], axis=1))

        if self.motion_energy < IDLE_MOTION_THRESH: self.idle_frame_counter+=1
        else: self.idle_frame_counter=0
        self.status = "Idle" if self.idle_frame_counter > self.idle_time_thresh_frames else "Active"
        
        if self.status == "Active":
            ankle_kpt=get_keypoint(kpts_unnorm, KP_L_AN)
            if ankle_kpt[2]>0.1: self.zone=get_court_zone(ankle_kpt[1],roi,frame_h)

            # --- FIX: Robust Dominant Arm Detection ---
            s, e, w = KP_L_SH, KP_L_EL, KP_L_WR # Default to left arm
            if self.prev_kpts is not None:
                # Calculate motion for each arm
                left_arm_motion = np.sum(np.linalg.norm(kpts[LEFT_ARM_INDICES,:2].cpu() - self.prev_kpts[LEFT_ARM_INDICES,:2].cpu(), axis=1))
                right_arm_motion = np.sum(np.linalg.norm(kpts[RIGHT_ARM_INDICES,:2].cpu() - self.prev_kpts[RIGHT_ARM_INDICES,:2].cpu(), axis=1))
                if right_arm_motion > left_arm_motion: # Right arm moved more
                    s, e, w = KP_R_SH, KP_R_EL, KP_R_WR
            
            c_angle=calculate_angle(get_keypoint(kpts_unnorm,s),get_keypoint(kpts_unnorm,e),get_keypoint(kpts_unnorm,w))
            self.pose_velocity=abs(c_angle-self.prev_elbow_angle) if c_angle is not None and self.prev_elbow_angle is not None else 0.0
            self.prev_elbow_angle = c_angle
            
            self.motion_energy_buffer.append(self.motion_energy)
            kpts_norm=kpts[:,:2].cpu().numpy(); kpts_norm[:,0]/=frame_w; kpts_norm[:,1]/=frame_h
            self.keypoint_buffer.append(kpts_norm)

        self.prev_kpts = kpts.clone()

    def check_trigger(self):
        if self.status=="Idle" or self.cooldown_timer>0 or len(self.motion_energy_buffer)<MOTION_PEAK_WINDOW: return False,0,""
        energies=list(self.motion_energy_buffer)
        peak_idx=MOTION_PEAK_WINDOW//2
        is_peak=energies[peak_idx]>0.1 and energies[peak_idx]>=max(energies)
        if not is_peak: return False,0,""
        peak_energy=energies[peak_idx]
        if self.zone=="BACK" and peak_energy>BACK_COURT_MOTION_THRESH:
            self.cooldown_timer=int(15+peak_energy/5); return True,peak_energy,"Back Court Motion"
        if self.zone=="NET" and self.pose_velocity>NET_ZONE_POSE_VELOCITY_THRESH:
            self.cooldown_timer=int(15+peak_energy/5); return True,self.pose_velocity,"Net Zone Velocity"
        return False,0,""

    def set_last_shot(self,name): self.last_shot=name; self.last_shot_countdown=int(self.video_fps*2.5)
    def tick(self):
        if self.cooldown_timer>0: self.cooldown_timer-=1
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
    out=cv2.VideoWriter(f"{os.path.splitext(os.path.basename(opt.source))[0]}_debug_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    
    player_states={}; frame_num=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret: break
        frame_num+=1
        annotated_frame=frame.copy()
        if roi: cv2.rectangle(annotated_frame,(roi[0],roi[1]),(roi[0]+roi[2],roi[1]+roi[3]),(0,255,0),1)

        for state in player_states.values(): state.tick()

        results=pose_model.track(frame,persist=True,verbose=False,device=device)
        active_ids=set()
        if results and results[0].boxes and results[0].boxes.id is not None:
            track_ids,kpts_data,kpts_unnorm=results[0].boxes.id.int().cpu().tolist(),results[0].keypoints.data,results[0].keypoints.cpu().numpy()
            boxes_xyxy=results[0].boxes.xyxy.cpu().numpy()
            for i,track_id in enumerate(track_ids):
                active_ids.add(track_id)
                box=boxes_xyxy[i]
                if roi and not (box[0]<roi[0]+roi[2] and box[2]>roi[0] and box[1]<roi[1]+roi[3] and box[3]>roi[1]): continue
                if track_id not in player_states: player_states[track_id]=PlayerState(track_id,fps)
                state=player_states[track_id]
                state.update_kinematics(kpts_data[i],kpts_unnorm[i],(w,h),roi)
                
                if VERBOSE_LOG and state.status=='Active':
                    buffer_str=', '.join([f'{e:.1f}' for e in state.motion_energy_buffer])
                    print(f"F:{frame_num}|P:{track_id}|S:{state.status}|Z:{state.zone}|M:{state.motion_energy:>4.1f}|V:{state.pose_velocity:>4.1f}|Cooldown:{state.cooldown_timer}|Buffer:[{buffer_str}]")

                is_triggered,trigger_val,trigger_type=state.check_trigger()
                if is_triggered and len(state.keypoint_buffer)==SEQUENCE_LENGTH:
                    pose_video=create_pose_video_from_kpts(list(state.keypoint_buffer),SEQUENCE_LENGTH,NUM_KEYPOINTS,HEATMAP_SIZE)
                    input_tensor=torch.tensor(pose_video).unsqueeze(0).to(device)
                    with torch.no_grad(): shot_name=idx_to_class[torch.argmax(classifier(input_tensor)).item()]
                    state.set_last_shot(shot_name.replace('_',' '))
                    thresh=BACK_COURT_MOTION_THRESH if trigger_type=="Back Court Motion" else NET_ZONE_POSE_VELOCITY_THRESH
                    log_successful_shot(frame_num,state,trigger_type,trigger_val,thresh,shot_name)

                x1,y1=int(box[0]),int(box[1])
                color=(0,255,0) if state.status=='Active' else (0,255,255)
                cv2.rectangle(annotated_frame,(x1,y1),(int(box[2]),int(box[3])),color,1)
                cv2.putText(annotated_frame,f"P{track_id} {state.status}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                if state.last_shot_countdown>0: cv2.putText(annotated_frame,state.last_shot,(x1,y1-25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        
        inactive_ids=set(player_states.keys())-active_ids
        for inactive_id in inactive_ids: del player_states[inactive_id]

        out.write(annotated_frame); cv2.imshow("AI Coach Debugger",annotated_frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_coach(parse_opt())