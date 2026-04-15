import {
    FaceDetector,
    FaceLandmarker,
    FilesetResolver,
    ImageClassifier,
    ImageEmbedder
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

window.FaceDetector = FaceDetector;
window.FaceLandmarker = FaceLandmarker;
window.ImageClassifier = ImageClassifier;
window.ImageEmbedder = ImageEmbedder;
window.FilesetResolver = FilesetResolver;
window.dispatchEvent(new Event('mp-tasks-ready'));