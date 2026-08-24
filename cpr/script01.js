// 1. Import necessary MediaPipe classes
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');

// --- Configuration ---
const VIDEO_SOURCE = '02_pose_raw_correct_video_front240626-07.mp4';

// Sigmoid threshold: Raw Score >= CONFIDENCE_THRESHOLD -> Correct (1), else Incorrect (0)
const CONFIDENCE_THRESHOLD = 0.002; 

const classLabels = {
    0: 'Incorrect',
    1: 'Correct'
};

const classColors = {
    0: '#FF0000', // Red
    1: '#00FF00'  // Green
};

// --- STATE VARIABLES ---
let counter = 0;
let stage = "down"; 
let shoulderBaselineY = null;
let lastTimestamp = -1; 
let videoFrameCount = 0;

let poseLandmarker;
let classifierModel;
let drawingUtils;
let upperBodyConnections = [];

async function setupApp() {
    try {
        drawingUtils = new DrawingUtils(canvasCtx);
        
        // Filter MediaPipe pose connections to upper body only (landmarks 0 through 24)
        upperBodyConnections = PoseLandmarker.POSE_CONNECTIONS.filter(
            conn => (conn.start ?? conn[0]) <= 24 && (conn.end ?? conn[1]) <= 24
        );

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        if (window.tflite && tflite.setWasmPath) {
            tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');
        }

        // Load cprmodel.tflite
        classifierModel = await tflite.loadTFLiteModel('./newcprmodel.tflite');

        video.muted = true;
        video.loop = true;
        video.playsInline = true;

        video.addEventListener('loadeddata', async () => {
            loadingOverlay.classList.add('hidden');
            try {
                await video.play();
                renderLoop();
            } catch (err) {
                console.warn("Autoplay blocked. Click video to play:", err);
                renderLoop(); 
            }
        });

        video.addEventListener('error', (e) => {
            console.error("Video failed to load:", VIDEO_SOURCE, e);
            loadingOverlay.innerHTML = `
                <p style="color: #FF5555; font-weight: bold;">Failed to load video: ${VIDEO_SOURCE}</p>
                <p>Select your MP4 file manually:</p>
                <input type="file" id="video-picker" accept="video/mp4,video/*">
            `;
            
            document.getElementById('video-picker')?.addEventListener('change', (evt) => {
                const file = evt.target.files[0];
                if (file) {
                    video.src = URL.createObjectURL(file);
                    loadingOverlay.classList.add('hidden');
                    videoFrameCount = 0;
                    video.load();
                }
            });
        });

        video.src = VIDEO_SOURCE;
        video.load();

    } catch (error) {
        console.error("Initialization failed:", error);
    }
}

async function renderLoop() {
    if (video.paused || video.ended) {
        requestAnimationFrame(renderLoop);
        return;
    }

    try {
        let startTimeMs = performance.now();
        if (startTimeMs <= lastTimestamp) {
            startTimeMs = lastTimestamp + 1;
        }
        lastTimestamp = startTimeMs;

        const results = poseLandmarker.detectForVideo(video, startTimeMs);

        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (results && results.landmarks && results.landmarks.length > 0) {
            videoFrameCount++;
            const landmarks = results.landmarks[0];

            // Isolate upper body landmarks (0 through 24)
            const upperBodyLandmarks = landmarks.slice(0, 25);

            // Draw skeleton
            drawingUtils.drawConnectors(landmarks, upperBodyConnections, { color: "#FFFFFF", lineWidth: 2 });
            drawingUtils.drawLandmarks(upperBodyLandmarks, { color: "#00FF00", lineWidth: 1, radius: 3 });

            // --- 1. Binary CPR Pose Classification --- 
            let predictedIndex = 0;
            let rawScore = 0.0;

            if (classifierModel) {
                try {
                    // Mirror X coordinate (1.0 - x) to match save_to_csv.py cv2.flip(image, 1)
                    const landmarkData = upperBodyLandmarks.flatMap(lm => [
                        1.0 - lm.x, 
                        lm.y, 
                        lm.z, 
                        lm.visibility ?? 0
                    ]); 

                    // Include frame index as feature 0 matching CSV dataset
                    const flattened = [videoFrameCount, ...landmarkData]; 

                    const inputTensor = tf.tensor2d([landmarkData], [1, 100]); 
                    const output = classifierModel.predict(inputTensor);

                    let outputTensor = null;
                    if (output instanceof tf.Tensor) {
                        outputTensor = output;
                    } else if (output && typeof output === 'object') {
                        const keys = Object.keys(output);
                        if (keys.length > 0) outputTensor = output[keys[0]];
                    }

                    if (outputTensor && typeof outputTensor.dataSync === 'function') {
                        rawScore = outputTensor.dataSync()[0];
                        
                        // Compare raw output against threshold
                        predictedIndex = rawScore >= CONFIDENCE_THRESHOLD ? 1 : 0; 
                    }

                    inputTensor.dispose();
                    if (output instanceof tf.Tensor) {
                        output.dispose();
                    } else if (output && typeof output === 'object') {
                        Object.values(output).forEach(t => t?.dispose?.());
                    }
                } catch (tfErr) {
                    console.error("Tensor Classification Error:", tfErr);
                }
            }

            // --- 2. SHOULDER-ONLY CYCLE TRACKER ---
            const leftShoulderY = landmarks[11].y;
            const rightShoulderY = landmarks[12].y;
            const currentShoulderY = (leftShoulderY + rightShoulderY) / 2;

            const noseY = landmarks[0].y;
            const leftEyeY = landmarks[2].y;
            const headScale = Math.abs(noseY - leftEyeY); 
            const MIN_MOVEMENT = headScale * 0.6; 

            if (shoulderBaselineY === null) {
                shoulderBaselineY = currentShoulderY;
            }

            const travelDistance = shoulderBaselineY - currentShoulderY;

            if (travelDistance > MIN_MOVEMENT && stage === "down") {
                stage = "up";
                shoulderBaselineY = currentShoulderY; 
            }
            
            if (currentShoulderY > (shoulderBaselineY + MIN_MOVEMENT) && stage === "up") {
                stage = "down";
                counter += 1; 
                shoulderBaselineY = currentShoulderY; 
            }

            shoulderBaselineY = shoulderBaselineY * 0.98 + currentShoulderY * 0.02;

            drawUI(
                classLabels[predictedIndex] || 'Unknown', 
                classColors[predictedIndex] || '#FFFFFF', 
                rawScore, 
                counter, 
                stage
            );
        }

        canvasCtx.restore();
    } catch (error) {
        console.error("Render loop error detail:", error);
    } finally {
        requestAnimationFrame(renderLoop);
    }
}

function drawUI(label, color, rawScore, count, currentStage) {
    canvasCtx.save();
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);

    // Dark semi-transparent background HUD box
    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.75)";
    canvasCtx.fillRect(20, 20, 360, 185);

    // Pose Label
    canvasCtx.font = "bold 26px Arial";
    canvasCtx.fillStyle = color;
    canvasCtx.fillText(`POSE: ${label}`, 35, 55);

    // Raw Model Output Score & Threshold Display
    canvasCtx.font = "18px Arial";
    canvasCtx.fillStyle = "#00FFFF";
    canvasCtx.fillText(`Raw Model Score: ${rawScore.toFixed(4)}`, 35, 85);
    
    canvasCtx.fillStyle = "#FFD700";
    canvasCtx.fillText(`Active Threshold: ${CONFIDENCE_THRESHOLD.toFixed(2)}`, 35, 110);

    // Reps Display
    canvasCtx.font = "bold 22px Arial";
    canvasCtx.fillStyle = "#FFFFFF";
    canvasCtx.fillText(`REPS: ${count}`, 35, 145);
    
    // Movement Stage Display
    canvasCtx.fillStyle = currentStage === "up" ? "#00FF00" : "#FF0000";
    canvasCtx.fillText(`STAGE: ${currentStage.toUpperCase()}`, 35, 175);

    canvasCtx.restore();
}

setupApp();