// 1. Import necessary MediaPipe classes
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const loadingOverlay = document.getElementById('loading-overlay');

// --- Configuration ---
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

// Scaler variables
let scalerMean = [];
let scalerStd = [];

// Rate tracking variables (30-second sliding window)
let repTimestamps = [];
let currentCPM = 0;

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

        // Fetch StandardScaler parameters from Python
        try {
            const scalerRes = await fetch('./scaler_params.json');
            const scalerData = await scalerRes.json();
            scalerMean = scalerData.mean;
            scalerStd = scalerData.std;
            console.log("StandardScaler parameters loaded successfully.");
        } catch (scalerErr) {
            console.error("Could not load scaler_params.json:", scalerErr);
        }

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
        video.playsInline = true;

        // --- WEBCAM STREAM SETUP ---
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });

            video.srcObject = stream;

            video.onloadedmetadata = () => {
                video.play();
                loadingOverlay.classList.add('hidden');
                renderLoop();
            };
        } catch (webcamErr) {
            console.error("Error accessing webcam:", webcamErr);
            loadingOverlay.innerHTML = `
                <p style="color: #FF5555; font-weight: bold; text-align: center;">
                    Webcam access denied or unavailable.<br>
                    Please grant camera permissions and reload.
                </p>
            `;
        }

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

        // Update 30-second sliding window rate calculation
        const now = performance.now();
        repTimestamps = repTimestamps.filter(t => (now - t) <= 30000); 
        currentCPM = repTimestamps.length * 2; 

        // --- DRAW CPR CENTER GUIDE BOX ---
        drawCPRGuideBox();

        if (results && results.landmarks && results.landmarks.length > 0) {
            videoFrameCount++;
            const landmarks = results.landmarks[0];

            // Isolate upper body landmarks (0 through 24)
            const upperBodyLandmarks = landmarks.slice(0, 25);

            // Draw skeleton
            drawingUtils.drawConnectors(landmarks, upperBodyConnections, { color: "#FFFFFF", lineWidth: 2 });
            drawingUtils.drawLandmarks(upperBodyLandmarks, { color: "#00FF00", lineWidth: 1, radius: 3 });

            // --- 1. Softmax (2-Class) CPR Pose Classification --- 
            let predictedIndex = 0;
            let confidenceScore = 0.0;

            if (classifierModel) {
                try {
                    // Extract 100 raw landmark features [x, y, z, visibility]
                    const rawLandmarks = upperBodyLandmarks.flatMap(lm => [
                        1.0 - lm.x, 
                        lm.y, 
                        lm.z, 
                        lm.visibility ?? 0
                    ]);

                    // APPLY STANDARD SCALER: (val - mean) / std
                    const scaledLandmarks = rawLandmarks.map((val, idx) => {
                        if (scalerMean.length > idx && scalerStd.length > idx && scalerStd[idx] !== 0) {
                            return (val - scalerMean[idx]) / scalerStd[idx];
                        }
                        return val;
                    });

                    const inputTensor = tf.tensor2d([scaledLandmarks], [1, 100]); 
                    const output = classifierModel.predict(inputTensor);

                    let outputTensor = null;
                    if (output instanceof tf.Tensor) {
                        outputTensor = output;
                    } else if (output && typeof output === 'object') {
                        const keys = Object.keys(output);
                        if (keys.length > 0) outputTensor = output[keys[0]];
                    }

                    if (outputTensor && typeof outputTensor.dataSync === 'function') {
                        const probabilities = outputTensor.dataSync(); // Array of [prob_0, prob_1]
                        
                        if (probabilities.length >= 2) {
                            predictedIndex = probabilities[1] > probabilities[0] ? 1 : 0;
                            confidenceScore = probabilities[predictedIndex];
                        } else {
                            predictedIndex = probabilities[0] >= 0.5 ? 1 : 0;
                            confidenceScore = probabilities[0];
                        }
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
                
                repTimestamps.push(performance.now());
            }

            shoulderBaselineY = shoulderBaselineY * 0.98 + currentShoulderY * 0.02;

            drawUI(
                classLabels[predictedIndex] || 'Unknown', 
                classColors[predictedIndex] || '#FFFFFF', 
                confidenceScore, 
                counter, 
                stage,
                currentCPM
            );
        }

        canvasCtx.restore();
    } catch (error) {
        console.error("Render loop error detail:", error);
    } finally {
        requestAnimationFrame(renderLoop);
    }
}

function drawCPRGuideBox() {
    canvasCtx.save();

    const boxWidth = 450; 
    const boxHeight = 850; 

    const x = ((canvasElement.width - boxWidth) / 2) + 70;
    const y = (canvasElement.height - boxHeight) / 2;

    canvasCtx.fillStyle = "rgba(0, 255, 255, 0.15)"; 
    canvasCtx.fillRect(x, y, boxWidth, boxHeight);

    canvasCtx.strokeStyle = "#00FFFF";
    canvasCtx.lineWidth = 3;
    canvasCtx.setLineDash([10, 6]);
    canvasCtx.strokeRect(x, y, boxWidth, boxHeight);

    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);

    canvasCtx.font = "bold 18px Arial";
    canvasCtx.fillStyle = "#FFFFFF";
    canvasCtx.textAlign = "center";
    canvasCtx.textBaseline = "middle";
    canvasCtx.fillText("ALIGN CHEST HERE", (canvasElement.width / 2) - 50, ((canvasElement.height / 2)) - 120);

    canvasCtx.restore();
}

function drawUI(label, color, confidenceScore, count, currentStage, rateCPM) {
    canvasCtx.save();
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);

    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.75)";
    canvasCtx.fillRect(20, 20, 380, 220);

    canvasCtx.font = "bold 26px Arial";
    canvasCtx.fillStyle = color;
    canvasCtx.fillText(`POSE: ${label}`, 35, 55);

    canvasCtx.font = "18px Arial";
    canvasCtx.fillStyle = "#00FFFF";
    canvasCtx.fillText(`Confidence: ${(confidenceScore * 100).toFixed(1)}%`, 35, 85);

    canvasCtx.font = "bold 22px Arial";
    canvasCtx.fillStyle = "#FFFFFF";
    canvasCtx.fillText(`REPS: ${count}`, 35, 145);
    
    canvasCtx.fillStyle = currentStage === "up" ? "#00FF00" : "#FF0000";
    canvasCtx.fillText(`STAGE: ${currentStage.toUpperCase()}`, 35, 175);

    let rateColor = "#FFD700"; 
    if (rateCPM >= 100 && rateCPM <= 120) {
        rateColor = "#00FF00"; 
    } else if (rateCPM > 120) {
        rateColor = "#FF5555"; 
    }

    canvasCtx.font = "bold 20px Arial";
    canvasCtx.fillStyle = rateColor;
    canvasCtx.fillText(`RATE: ${rateCPM} CPM (Target: 100-120)`, 35, 210);

    canvasCtx.restore();
}

setupApp();