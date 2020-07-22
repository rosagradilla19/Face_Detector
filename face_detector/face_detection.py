import cv2
import torch
import numpy as np

class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn, classifier, mode):
        self.mtcnn = mtcnn
        self.classifier = classifier
        self.mode = mode

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def _detect_ROI(self, boxes):
        """
        Return ROIs as a list with each element being the coordinates for face detection box
        (RowStart, RowEnd, ColStart, ColEnd)
        """
        ROIs = list()
        for box in boxes:
            ROI = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            ROIs.append(ROI)
        return ROIs

    def _blur_face(self, image, factor=3.0):
        """ 
        Blur function to be applied to ROIs
        """
        # Determine size of blurring kernel based on input image
        (h, w) = image.shape[:2]
        kW = int(w / factor)
        kH = int(h / factor)

        # Ensure width and height of kernel are odd
        if kW % 2 == 0:
            kW -= 1

        if kH % 2 == 0:
            kH -= 1

        # Apply a Gaussian blur to the input image using our computed kernel size
        return cv2.GaussianBlur(image, (kW, kH), 0)
    
    def _is_it_Rosa(self, face):
        """
        This function takes the face (as a numpy array), runs the classifier
        on the array and returns wether it is Rosa or not
        """
        # Turn the face to grayscale for inference
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Resize to 160x160 for inference
        image = cv2.resize(gray, (160,160))
        # Convert to pytorch tensor and normalize
        tensor_ = torch.from_numpy(image).view(-1,1,160,160)
        tensor_ = tensor_/255.0
        # run classifier on face
        with torch.no_grad():
            out = self.classifier(tensor_)
        prediction = torch.argmax(out).numpy()
        if prediction == 0:
            return('not_Rosa')
        else:
            return('Rosa')

    def _is_it_Rosa_tf(self, face):
        """
        This function takes the face (as a numpy array), runs the classifier
        on the array and returns wether it is Rosa or not
        """
        # Resize to 128x128 for inference
        image = cv2.resize(face, (128,128))
        # Convert to pytorch tensor and normalize
        tensor_ = torch.from_numpy(image).view(-1,3,128,128)
        tensor_ = tensor_/255.0
        # run classifier on face
        with torch.no_grad():
            out = self.classifier(tensor_)
        _, prediction = torch.max(out, 1)
        prediction = np.array(prediction[0])
        if prediction == 0:
            return('Rosa')
        else:
            return('not_Rosa')

    def run(self, blur_setting=True):

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            try:
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                self._draw(frame, boxes, probs, landmarks)

                # Blur faces in the frame
                if blur_setting == True:

                        #Extract face ROIs
                    ROIs = self._detect_ROI(boxes)

                    for roi in ROIs:
                        (startY, endY, startX, endX) = roi
                        face = frame[startY:endY, startX:endX]
                        # check for mode
                        if self.mode == 'tl_model':
                            prediction = self._is_it_Rosa_tf(face)
                        if self.mode == 'simple_cnn':
                            prediction = self._is_it_Rosa(face)

                        if prediction == 'Rosa':
                            face = self._blur_face(face)
                            frame[startY:endY, startX:endX] = face
                        elif prediction == 'not_Rosa':
                            pass

            except:
                pass

            # Show the frame
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
