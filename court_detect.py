import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as F

# -----------------------------------------------------------------------------
# ## MODEL DEFINITION (Copied from your training script) ##
# -----------------------------------------------------------------------------
def get_model(num_keypoints=22, num_classes=2):
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_features_kp, num_keypoints)
    return model

# -----------------------------------------------------------------------------

class CourtDetect(object):
    def __init__(self, model_path='models/best_model.pth', confidence_threshold=0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.threshold = confidence_threshold
        self.keypoints = None
        self.setup_RCNN()

    def setup_RCNN(self):
        """
        Correctly loads the trained model from a checkpoint file.
        """
        # 1. Create an instance of the model architecture with 22 keypoints
        self.model = get_model(num_keypoints=22, num_classes=2)

        # 2. Load the entire checkpoint dictionary
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # 3. Load the weights from the checkpoint into the model instance
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 4. Move the model to the correct device and set to evaluation mode
        self.model.to(self.device).eval()
        print("Court detection model loaded successfully.")

    def detect_court(self, frame):
        """
        Detects the court in a single frame and stores the keypoints.
        Returns True if a court is found, False otherwise.
        """
        # Prepare the frame for the model
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = F.to_tensor(image_pil).unsqueeze(0).to(self.device)

        # Get model predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Find the prediction with the highest score
        if len(predictions[0]['scores']) == 0:
            self.keypoints = None
            return False

        top_score_index = torch.argmax(predictions[0]['scores'])
        top_score = predictions[0]['scores'][top_score_index]

        if top_score > self.threshold:
            self.keypoints = predictions[0]['keypoints'][top_score_index].cpu().numpy()
            return True
        
        self.keypoints = None
        return False

    def get_court_corners(self):
        """
        Returns the four outer corners of the court from the detected keypoints.
        This is the primary function needed by the preprocessing script.
        """
        if self.keypoints is None:
            return None

        # IMPORTANT: The indices [0, 5, 16, 21] are an assumption for the
        # four outer corners of a 22-point court layout. You may need to
        # visualize your keypoints and adjust these indices to match your dataset's annotation scheme.
        # 
        try:
            top_left = self.keypoints[0][:2]
            top_right = self.keypoints[4][:2]
            bottom_left = self.keypoints[17][:2]
            bottom_right = self.keypoints[21][:2]
            return [top_left, top_right, bottom_left, bottom_right]
        except IndexError:
            print("Warning: Keypoint indices for corners are out of bounds. Using None.")
            return None

    def draw_detected_court(self, frame):
        """
        Draws all detected keypoints on a frame for visualization.
        """
        if self.keypoints is None:
            return frame
        
        output_frame = frame.copy()
        for i, (x, y, v) in enumerate(self.keypoints):
            if v > 0: # v=1: hidden, v=2: visible
                cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(output_frame, str(i), (int(x) + 5, int(y) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return output_frame