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

// Rate tracking variables (30-second sliding window)
let repTimestamps = [];
let currentCPM = 0;

// Compression tracking only begins once the correct pose is detected
let compressionStarted = false;
let feedback = { text: "", color: "#FFFFFF" };
const MIN_GOOD_CPM = 100;
const MAX_GOOD_CPM = 120;
const RATE_WINDOW_REPS = 6;   // use the last N reps to estimate rate
const MIN_REPS_FOR_FEEDBACK = 3;

// CSV RECORDING VARIABLES
let isRecording = false;
let csvRows = [];

let poseLandmarker;
let classifierModel;
let drawingUtils;
let upperBodyConnections = [];

// --- CSV UI CONTROLS ---
function setupCSVControls() {
    const controlsDiv = document.createElement('div');
    controlsDiv.style.cssText = 'position: absolute; top: 10px; right: 10px; z-index: 1000; display: flex; gap: 10px;';

    const recordBtn = document.createElement('button');
    recordBtn.innerText = 'Start Recording CSV';
    recordBtn.style.cssText = 'padding: 10px 15px; background: #00FF00; border: none; font-weight: bold; cursor: pointer; border-radius: 5px;';

    const downloadBtn = document.createElement('button');
    downloadBtn.innerText = 'Download CSV';
    downloadBtn.style.cssText = 'padding: 10px 15px; background: #00FFFF; border: none; font-weight: bold; cursor: pointer; border-radius: 5px;';
    downloadBtn.disabled = true;

    recordBtn.onclick = () => {
        isRecording = !isRecording;
        if (isRecording) {
            csvRows = [];
            // Header for 100 landmark attributes (25 landmarks * 4 attributes)
            const headers = [];
            for (let i = 0; i < 25; i++) {
                headers.push(`lm${i}_x`, `lm${i}_y`, `lm${i}_z`, `lm${i}_vis`);
            }
            csvRows.push(headers.join(','));
            
            recordBtn.innerText = 'Stop Recording';
            recordBtn.style.background = '#FF0000';
            recordBtn.style.color = '#FFFFFF';
            downloadBtn.disabled = true;
        } else {
            recordBtn.innerText = 'Start Recording CSV';
            recordBtn.style.background = '#00FF00';
            recordBtn.style.color = '#000000';
            downloadBtn.disabled = csvRows.length <= 1;
        }
    };

    downloadBtn.onclick = () => {
        if (csvRows.length <= 1) return;
        const csvContent = "data:text/csv;charset=utf-8," + csvRows.join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `cpr_upperbody_datapoints_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    controlsDiv.appendChild(recordBtn);
    controlsDiv.appendChild(downloadBtn);
    document.body.appendChild(controlsDiv);
}

async function setupApp() {
    try {
        setupCSVControls();
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
            tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/');
        }

        // Load newcprmodel.tflite (cache-busted to avoid loading a stale cached model)
        classifierModel = await tflite.loadTFLiteModel(`newcprmodel.tflite?v=${Date.now()}`);

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
        currentCPM = calculateCPM(now);

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
                    // Extract 100 raw landmark features [x, y, z, visibility] - no mirror, no scaling
                    const rawLandmarks = upperBodyLandmarks.flatMap(lm => [
                        lm.x, 
                        lm.y, 
                        lm.z, 
                        lm.visibility ?? 0
                    ]);

                    const inputTensor = tf.tensor2d([rawLandmarks], [1, 100]); 
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
                        console.log("Model output length:", probabilities.length, Array.from(probabilities)); // TEMP DEBUG - remove once confirmed working
                        
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

            // --- 2. SHOULDER-ONLY CYCLE TRACKER (starts on correct pose) ---
            const poseIsCorrect = predictedIndex === 1;

            if (poseIsCorrect && !compressionStarted) {
                // Correct pose detected for the first time: start a fresh session
                compressionStarted = true;
                counter = 0;
                stage = "down";
                shoulderBaselineY = null;
                repTimestamps = [];
                currentCPM = 0;
            }

            if (compressionStarted) {
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
                    shoulderBaselineY = currentShoulderY; 
                    // Only count the compression if the pose is currently correct
                    if (poseIsCorrect) {
                        counter += 1; 
                        repTimestamps.push(performance.now());
                    }
                }

                shoulderBaselineY = shoulderBaselineY * 0.98 + currentShoulderY * 0.02;
            }

            updateFeedback(poseIsCorrect, performance.now());

            // --- RECORD EXACTLY 100 LANDMARK DATAPOINTS ---
            if (isRecording) {
                // 100 features: 25 landmarks * 4 values (x, y, z, visibility)
                const landmarkValues = upperBodyLandmarks.flatMap(lm => [
                    lm.x.toFixed(5),
                    lm.y.toFixed(5),
                    lm.z.toFixed(5),
                    (lm.visibility ?? 0).toFixed(5)
                ]);

                csvRows.push(landmarkValues.join(','));
            }

            drawUI(
                classLabels[predictedIndex] || 'Unknown', 
                classColors[predictedIndex] || '#FFFFFF',  
                confidenceScore, 
                counter, 
                stage,
                currentCPM
            );
            drawFeedbackBanner();
        }

        canvasCtx.restore();
    } catch (error) {
        console.error("Render loop error detail:", error);
    } finally {
        requestAnimationFrame(renderLoop);
    }
}

function calculateCPM(now) {
    if (!compressionStarted || repTimestamps.length < 2) return 0;

    // Estimate rate from the most recent reps
    const recent = repTimestamps.slice(-RATE_WINDOW_REPS);
    const spanMs = recent[recent.length - 1] - recent[0];
    if (spanMs <= 0) return 0;
    let cpm = ((recent.length - 1) / spanMs) * 60000;

    // If the user has stopped/slowed down since the last rep, let the rate decay
    const sinceLast = now - recent[recent.length - 1];
    const currentInterval = 60000 / Math.max(cpm, 1);
    if (sinceLast > currentInterval) {
        cpm = Math.min(cpm, 60000 / sinceLast);
    }
    return Math.round(cpm);
}

function updateFeedback(poseIsCorrect, now) {
    if (!compressionStarted) {
        feedback = { text: "Get into the correct pose to begin", color: "#FFD700" };
        return;
    }
    if (!poseIsCorrect) {
        feedback = { text: "Correct your pose to keep counting", color: "#FF5555" };
        return;
    }
    if (repTimestamps.length < MIN_REPS_FOR_FEEDBACK) {
        feedback = { text: "Start compressions...", color: "#FFFFFF" };
        return;
    }

    // Stalled for over 1.5s: definitely too slow
    const sinceLast = now - repTimestamps[repTimestamps.length - 1];
    if (currentCPM < MIN_GOOD_CPM || sinceLast > 1500) {
        feedback = { text: "GO FASTER!", color: "#FFA500" };
    } else if (currentCPM <= MAX_GOOD_CPM) {
        feedback = { text: "Congratulations! Great compression rate!", color: "#00FF00" };
    } else {
        feedback = { text: "SLOW DOWN a little", color: "#FF5555" };
    }
}

function drawFeedbackBanner() {
    if (!feedback.text) return;
    canvasCtx.save();
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvasElement.width, 0);

    const bannerW = Math.min(canvasElement.width - 40, 760);
    const bannerH = 70;
    const x = (canvasElement.width - bannerW) / 2;
    const y = canvasElement.height - bannerH - 30;

    canvasCtx.fillStyle = "rgba(0, 0, 0, 0.8)";
    canvasCtx.fillRect(x, y, bannerW, bannerH);
    canvasCtx.strokeStyle = feedback.color;
    canvasCtx.lineWidth = 4;
    canvasCtx.setLineDash([]);
    canvasCtx.strokeRect(x, y, bannerW, bannerH);

    canvasCtx.font = "bold 30px Arial";
    canvasCtx.fillStyle = feedback.color;
    canvasCtx.textAlign = "center";
    canvasCtx.textBaseline = "middle";
    canvasCtx.fillText(feedback.text, canvasElement.width / 2, y + bannerH / 2);
    canvasCtx.restore();
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