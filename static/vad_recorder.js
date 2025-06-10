// --- Backend URL Configuration ---
const BACKEND_WS_URL = "wss://pitchine-backend.onrender.com/ws";

// --- Element Selectors ---
const container = document.querySelector('.container');
const introSection = document.getElementById('introSection');
const actionBtn = document.getElementById('actionBtn');
const stopBtn = document.getElementById('stopBtn');
const practiceAgainBtn = document.getElementById('practiceAgainBtn');
const backToPracticeBtn = document.getElementById('backToPracticeBtn');
const statusDiv = document.getElementById('status');
const timerDiv = document.getElementById('timer');
const startupDetailsArea = document.getElementById('startupDetailsManagementArea');
const modeSelectionArea = document.getElementById('modeSelectionArea');
const pitchingArea = document.getElementById('pitchingArea');
const reportArea = document.getElementById('reportArea');
const reportContent = document.getElementById('reportContent');
const historyArea = document.getElementById('historyArea');
const userIdentifierInput = document.getElementById('userIdentifier');
const viewHistoryBtn = document.getElementById('viewHistoryBtn');
const sessionList = document.getElementById('sessionList');
const progressChartCanvas = document.getElementById('progressChart');
const investorAvatars = {
    "Alex Chen": document.getElementById('avatar-alex-chen'),
    "Maria Santos": document.getElementById('avatar-maria-santos'),
    "Ben Carter": document.getElementById('avatar-ben-carter')
};
const investorSpeechBubbles = {
    "Alex Chen": document.getElementById('speech-alex-chen'),
    "Maria Santos": document.getElementById('speech-maria-santos'),
    "Ben Carter": document.getElementById('speech-ben-carter')
};
const userTranscriptionDisplay = document.getElementById('user-transcription');
const fullLogDisplay = document.getElementById('transcription');
const authArea = document.getElementById('authArea');
const userNotLoggedInDiv = document.getElementById('userNotLoggedIn');
const userLoggedInDiv = document.getElementById('userLoggedIn');
const emailInput = document.getElementById('emailInput');
const passwordInput = document.getElementById('passwordInput');
const signUpBtn = document.getElementById('signUpBtn');
const loginBtn = document.getElementById('loginBtn');
const googleSignInBtn = document.getElementById('googleSignInBtn');
const logoutBtn = document.getElementById('logoutBtn');
const userEmailDisplay = document.getElementById('userEmailDisplay');
const userHistoryControls = document.getElementById('userHistoryControls');
const startupProfileSelect = document.getElementById('startupProfileSelect');
const manageProfilesBtn = document.getElementById('manageProfilesBtn');
const profilesModal = document.getElementById('profilesModal');
const closeProfilesModalBtn = document.getElementById('closeProfilesModal');
const profileListContainer = document.getElementById('profileListContainer');
const openNewProfileFormBtn = document.getElementById('openNewProfileFormBtn');
const profileFormContainer = document.getElementById('profileFormContainer');
const profileFormTitle = document.getElementById('profileFormTitle');
const editingProfileIdInput = document.getElementById('editingProfileId');
const modalStartupNameInput = document.getElementById('modalStartupName');
const modalOneLinerInput = document.getElementById('modalOneLiner');
const modalCoreProblemInput = document.getElementById('modalCoreProblem');
const saveProfileBtn = document.getElementById('saveProfileBtn');
const cancelProfileFormBtn = document.getElementById('cancelProfileFormBtn');
const mainStartupNameInput = document.getElementById('startupNameInput');
const mainOneLinerInput = document.getElementById('oneLinerInput');
const mainCoreProblemInput = document.getElementById('coreProblemInput');
// --- NEW SELECTORS ---
const responseControls = document.getElementById('responseControls');
const confirmBtn = document.getElementById('confirmBtn');
const resetBtn = document.getElementById('resetBtn');
const finishedSpeakingBtn = document.getElementById('finishedSpeakingBtn');


// --- Global State Variables ---
let socket;
let audioContext, mediaStream, mediaStreamSource, processorNode, mediaRecorder;
let pitchTimerInterval, pitchTimeout;
let responseTimeout;
let progressChart = null;
let appState = 'initial';
let currentSpeakingInvestor = null;
let investorExpectedToRespond = null;
let userHasSpokenInThisTurn = false;
let isAudioProcessingInitialized = false;
let isConnecting = false;
let masterRecorder;
let fullAudioChunks = [];
let conversationHistory = [];
let pitchStartTime;
let sessionId;
let savedStartupProfiles = [];
let currentSelectedProfileData = null;
let utteranceChunks = [];

// --- Firebase Config & Initialization ---
const firebaseConfig = {
  apiKey: "AIzaSyB-HLZWxE7y4FwunHvEw-BH_IkOr1H0SpI",
  authDomain: "pitchine-ed6c2.firebaseapp.com",
  projectId: "pitchine-ed6c2",
  storageBucket: "pitchine-ed6c2.appspot.com",
  messagingSenderId: "323898270266",
  appId: "1:323898270266:web:d4edd9b54082db17c68b03",
  measurementId: "G-9R50037REV"
};
firebase.initializeApp(firebaseConfig);
firebase.analytics();
const fbAuth = firebase.auth();
const fbDb = firebase.firestore();
let currentUser = null;

// --- Constants ---
const VAD_ENERGY_THRESHOLD = 0.02;
const RESPONSE_TIMEOUT_MS = 30000;
const PITCH_DURATION_MS = 180000;

// --- Event Listeners ---
actionBtn.addEventListener('click', handleActionButtonClick);
stopBtn.addEventListener('click', endPitchSession);
practiceAgainBtn.addEventListener('click', resetForPractice);
backToPracticeBtn.addEventListener('click', resetForPractice);
viewHistoryBtn.addEventListener('click', handleViewHistoryClick);
signUpBtn.addEventListener('click', signUpUser);
loginBtn.addEventListener('click', loginUser);
googleSignInBtn.addEventListener('click', signInWithGoogle);
logoutBtn.addEventListener('click', logoutUser);
manageProfilesBtn.addEventListener('click', openManageProfilesModal);
closeProfilesModalBtn.addEventListener('click', () => profilesModal.style.display = "none");
openNewProfileFormBtn.addEventListener('click', () => {
    profileFormTitle.textContent = "Create New Profile";
    editingProfileIdInput.value = "";
    modalStartupNameInput.value = "";
    modalOneLinerInput.value = "";
    modalCoreProblemInput.value = "";
    profileFormContainer.style.display = "block";
});
saveProfileBtn.addEventListener('click', saveStartupProfile);
cancelProfileFormBtn.addEventListener('click', () => {
    profileFormContainer.style.display = "none";
});
startupProfileSelect.addEventListener('change', handleProfileSelectionChange);
// --- NEW EVENT LISTENERS ---
confirmBtn.addEventListener('click', handleConfirmResponse);
resetBtn.addEventListener('click', handleResetResponse);
finishedSpeakingBtn.addEventListener('click', handleFinishedSpeaking);

window.onclick = function(event) {
    if (event.target == profilesModal) {
        profilesModal.style.display = "none";
    }
}

// --- Core Logic ---

/**
 * Checks the URL on page load for a custom login token from the server.
 * If a token is found, it uses it to sign in to Firebase.
 */
function checkForLoginToken() {
    const urlParams = new URLSearchParams(window.location.search);
    const loginToken = urlParams.get('logintoken');
    const error = urlParams.get('error');

    if (error) {
        alert("Google Sign-In failed on the server. Please try again.");
        window.history.replaceState({}, document.title, "/");
    }

    if (loginToken) {
        fbAuth.signInWithCustomToken(loginToken)
            .then((userCredential) => {
                console.log("Signed in with custom token successfully!");
                window.history.replaceState({}, document.title, "/");
            })
            .catch((error) => {
                console.error("Error signing in with custom token:", error);
                alert("Could not complete sign-in. Error: " + error.message);
                window.history.replaceState({}, document.title, "/");
            });
    }
}

checkForLoginToken();

fbAuth.onAuthStateChanged(user => {
    console.log('Auth state changed:', user ? user.email : 'No user');
    if (user) {
        currentUser = user;
        userEmailDisplay.textContent = currentUser.email;
        introSection.style.display = 'none';
        userNotLoggedInDiv.style.display = 'none';
        userLoggedInDiv.style.display = 'block';
        userIdentifierInput.value = currentUser.email;
        userIdentifierInput.disabled = true;
        viewHistoryBtn.disabled = false;
        userHistoryControls.style.display = 'flex';

        if (BACKEND_WS_URL && BACKEND_WS_URL !== "wss://YOUR_RENDER_BACKEND_URL_HERE/ws") {
            if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
                 connectWebSocket();
            } else if (socket.readyState === WebSocket.OPEN) {
                updateStatus("Connected. Describe your startup to begin.");
                startupDetailsArea.style.display = 'block';
                modeSelectionArea.style.display = 'block';
                actionBtn.style.display = 'flex';
                updateButtonState('ready_to_pitch');
            }
        } else {
            updateStatus("Backend URL not configured. Please set BACKEND_WS_URL in the script.", true);
            updateButtonState('disconnected');
        }
        loadStartupProfiles();

    } else {
        currentUser = null;
        userEmailDisplay.textContent = '';
        introSection.style.display = 'block';
        userNotLoggedInDiv.style.display = 'flex';
        userLoggedInDiv.style.display = 'none';
        userIdentifierInput.value = '';
        userIdentifierInput.disabled = false;
        viewHistoryBtn.disabled = true;
        userHistoryControls.style.display = 'none';
        startupDetailsArea.style.display = 'none';
        modeSelectionArea.style.display = 'none';
        savedStartupProfiles = [];
        populateProfileDropdown();
        clearMainStartupInputs();
        if (appState !== 'initial') {
            performFullCleanupAndResetUI();
        }
        updateButtonState('initial');
        statusDiv.textContent = "Status: Please Login/Sign Up.";
    }
});

async function signUpUser() {
    const email = emailInput.value;
    const password = passwordInput.value;
    try {
        await fbAuth.createUserWithEmailAndPassword(email, password);
        alert("Sign up successful! You are now logged in.");
        passwordInput.value = '';
    } catch (error) {
        alert("Error signing up: " + error.message);
    }
}

async function loginUser() {
    const email = emailInput.value;
    const password = passwordInput.value;
    try {
        await fbAuth.signInWithEmailAndPassword(email, password);
        alert("Login successful!");
        passwordInput.value = '';
    } catch (error) {
        alert("Error logging in: " + error.message);
    }
}

function signInWithGoogle() {
    window.location.href = 'https://pitchine-backend.onrender.com/login/google';
}

async function logoutUser() {
    try {
        await fbAuth.signOut();
        alert("Error logging out: " + error.message);
    } catch (error){
        alert("Error logging out: " + error.message);
    }
}

function resetForPractice() {
    reportArea.style.display = 'none';
    historyArea.style.display = 'none';
    practiceAgainBtn.style.display = 'none';
    pitchingArea.style.display = 'none';
    timerDiv.style.display = 'none';
    fullLogDisplay.innerHTML = "";
    userTranscriptionDisplay.textContent = "";
    setActiveInvestor(null);
    responseControls.style.display = 'none';
    finishedSpeakingBtn.style.display = 'none';


    if (currentUser) {
        startupDetailsArea.style.display = 'block';
        modeSelectionArea.style.display = 'block';
        actionBtn.style.display = 'flex';

        if (socket && socket.readyState === WebSocket.OPEN) {
            updateButtonState('ready_to_pitch');
            statusDiv.textContent = "Status: Connected. Select or enter startup details.";
        } else {
            updateButtonState('disconnected');
            statusDiv.textContent = "Status: Not connected to backend.";
            if (BACKEND_WS_URL && BACKEND_WS_URL !== "wss://YOUR_RENDER_BACKEND_URL_HERE/ws" && (!socket || socket.readyState === WebSocket.CLOSED)) {
                connectWebSocket();
            }
        }
    } else {
        performFullCleanupAndResetUI();
    }
}

function updateButtonState(newState) {
    appState = newState;
    actionBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    responseControls.style.display = 'none';
    finishedSpeakingBtn.style.display = 'none';

    switch (newState) {
        case 'initial':
            actionBtn.textContent = 'Login/Sign Up to Practice';
            actionBtn.disabled = true;
            startupDetailsArea.style.display = 'none';
            modeSelectionArea.style.display = 'none';
            break;
        case 'disconnected':
             actionBtn.textContent = currentUser ? 'Connection Lost' : 'Login/Sign Up to Practice';
             actionBtn.disabled = true;
             startupDetailsArea.style.display = 'none';
             modeSelectionArea.style.display = 'none';
            break;
        case 'ready_to_pitch':
            actionBtn.textContent = 'Start Practice';
            actionBtn.disabled = false;
            startupDetailsArea.style.display = 'block';
            modeSelectionArea.style.display = 'block';
            break;
        case 'listening': // User's turn to speak
            actionBtn.style.display = 'none';
            stopBtn.style.display = 'flex';
            finishedSpeakingBtn.style.display = 'flex';
            userTranscriptionDisplay.textContent = "";
            userHasSpokenInThisTurn = false;
            break;
        case 'awaiting_confirmation': // User has finished speaking, awaiting confirm/reset
            actionBtn.style.display = 'none';
            stopBtn.style.display = 'flex';
            finishedSpeakingBtn.style.display = 'none';
            responseControls.style.display = 'flex';
            break;
        case 'processing':
             actionBtn.style.display = 'none';
             stopBtn.style.display = 'flex';
             finishedSpeakingBtn.style.display = 'none';
             responseControls.style.display = 'none';
             break;
        case 'generating_report':
            actionBtn.textContent = 'Generating Report...';
            actionBtn.disabled = true;
            actionBtn.style.display = 'flex';
            break;
        case 'report_complete':
        case 'history_view':
            actionBtn.style.display = 'none';
            break;
    }
}

async function ensureAudioContextIsRunning() {
    if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
    }
}

async function handleActionButtonClick() {
    await ensureAudioContextIsRunning();

    if (!currentUser && appState !== 'initial') {
        alert("Please log in to start practicing.");
        return;
    }
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        alert("Not connected to the backend. Please wait or check the connection.");
        return;
    }

    if (appState === 'ready_to_pitch') {
        const startupName = mainStartupNameInput.value.trim();
        const oneLiner = mainOneLinerInput.value.trim();
        const coreProblem = mainCoreProblemInput.value.trim();

        if (!startupName || !oneLiner || !coreProblem) {
            alert("Please select a startup profile or fill out all startup details.");
            return;
        }

        const pitchMode = document.querySelector('input[name="pitchMode"]:checked').value;
        const selectedProfileId = startupProfileSelect.value;

        socket.send(JSON.stringify({
            type: "startup_details",
            data: { name: startupName, pitch: oneLiner, problem: coreProblem, mode: pitchMode, profileId: selectedProfileId || null }
        }));
        if (selectedProfileId && currentUser) {
             fbDb.collection('users').doc(currentUser.uid).collection('startup_profiles').doc(selectedProfileId).update({
                lastUsedAt: firebase.firestore.FieldValue.serverTimestamp()
            }).catch(err => console.error("Error updating lastUsedAt:", err));
        }

        startupDetailsArea.style.display = 'none';
        modeSelectionArea.style.display = 'none';
        pitchingArea.style.display = 'flex';
        sessionId = `${currentUser.email.split('@')[0]}-${Date.now()}`;
        conversationHistory = [];
        pitchStartTime = Date.now();
        userHasSpokenInThisTurn = false;
        userTranscriptionDisplay.textContent = "";

        try {
            await initializeAudioProcessing();
            masterRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });
            fullAudioChunks = [];
            masterRecorder.ondataavailable = event => { if (event.data.size > 0) fullAudioChunks.push(event.data); };
            masterRecorder.start();

            if (pitchMode === 'strict') {
                updateStatus("Microphone active. Your 2-minute pitch session has started! Start speaking.", false);
                startPitchTimer();
                updateButtonState('listening');
            } else {
                updateStatus("Drill Mode activated. Waiting for the first question...", false);
                updateButtonState('processing'); // Wait for AI to start
            }
        } catch (err) {
            updateStatus(`Error accessing microphone: ${err.message}`, true);
            cleanupSessionButKeepConnection();
            resetForPractice();
        }
    }
}

// --- NEW/MODIFIED FUNCTIONS FOR CONFIRM/RESET ---

function handleFinishedSpeaking() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop(); // This will trigger mediaRecorder.onstop
    }
    updateButtonState('awaiting_confirmation'); // Transition to showing confirm/reset
    updateStatus("Review your transcription and confirm.", false);
}

function handleConfirmResponse() {
    if (appState !== 'awaiting_confirmation') return; // Check new state

    cancelResponseTimer(); // Clear any existing response timer

    const composedText = userTranscriptionDisplay.textContent.trim();
    const textToSend = composedText.length > 0 ? composedText : "[Silent Response]";

    appendToLog('You', textToSend);
    if(socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "send_composed_text", text: textToSend }));
    }

    userTranscriptionDisplay.textContent = "";
    userHasSpokenInThisTurn = false;
    updateButtonState('processing');
    updateStatus("Investor is thinking...", false);
}

function handleResetResponse() {
    if (appState !== 'awaiting_confirmation') return; // Check new state

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    }
    
    utteranceChunks = [];
    userTranscriptionDisplay.textContent = "";
    userHasSpokenInThisTurn = false;
    
    updateButtonState('listening');
    updateStatus("Your last response was cleared. Please try again.", false);
}


async function initializeAudioProcessing() {
    if (isAudioProcessingInitialized) return;
    try {
        if (!mediaStream || mediaStream.getTracks().every(track => track.readyState === 'ended')) {
             mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') await audioContext.resume();

        mediaStreamSource = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);

        processorNode.onaudioprocess = e => {
            // Only consider VAD if we are in 'listening' state and haven't already captured speech for this turn
            if (appState === 'listening' && !userHasSpokenInThisTurn) {
                const rms = Math.sqrt(e.inputBuffer.getChannelData(0).reduce((s, v) => s + v * v, 0) / e.inputBuffer.getChannelData(0).length);
                const isSpeaking = rms > VAD_ENERGY_THRESHOLD;

                if (isSpeaking) {
                    startInternalMediaRecorder(); // Start recording on first speech
                    userHasSpokenInThisTurn = true;
                }
            }
        };

        mediaStreamSource.connect(processorNode);
        processorNode.connect(audioContext.destination);
        isAudioProcessingInitialized = true;
    } catch (err) {
        updateStatus(`Error initializing audio processor: ${err.message}`, true);
        isAudioProcessingInitialized = false;
        throw err;
    }
}


async function endPitchSession() {
    if (appState === 'generating_report' || appState === 'disconnected' || appState === 'initial') return;
    updateStatus("Ending session... Uploading audio and generating report...");
    updateButtonState('generating_report');

    cleanupSessionMedia();

    if (masterRecorder?.state === "recording") {
        masterRecorder.onstop = async () => {
            await uploadAndFinalizeSession();
        };
        masterRecorder.stop();
    } else {
        await uploadAndFinalizeSession();
    }
}

async function uploadAndFinalizeSession() {
    const fullAudioBlob = new Blob(fullAudioChunks, { type: 'audio/webm' });
    const userIdentifierForSession = currentUser ? currentUser.email : "local_user";
    const currentSessionId = sessionId;
    let audioPath = "";

    if (socket?.readyState === WebSocket.OPEN && fullAudioBlob.size > 0) {
        const formData = new FormData();
        formData.append("file", fullAudioBlob, `${currentSessionId}.webm`);
        try {
            let httpUrl = BACKEND_WS_URL.replace('wss://', 'https://').replace('/ws', '');
            if (httpUrl.endsWith('/')) httpUrl = httpUrl.slice(0,-1);

            const response = await fetch(`${httpUrl}/upload-audio/${currentSessionId}`, { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok && result.status === 'success') {
                audioPath = result.path;
            } else { throw new Error(result.error || "Upload failed"); }
        } catch(e) {
            updateStatus("Error: Could not upload audio.", true);
            console.error("Audio upload error:", e);
        }
    }

    if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "end_session", session_id: currentSessionId, identifier: userIdentifierForSession, history: conversationHistory, audio_path: audioPath }));
    }
    fullAudioChunks = [];
}

function startInternalMediaRecorder() {
    if (!mediaStream) {
        console.error("MediaStream not available to start recorder.");
        return;
    }
    if (mediaRecorder?.state === "recording") {
        console.log("MediaRecorder is already recording.");
        return;
    }

    utteranceChunks = [];
    mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });

    mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            utteranceChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(utteranceChunks, { type: 'audio/webm;codecs=opus' });
        if (audioBlob.size > 0 && socket?.readyState === WebSocket.OPEN) {
            socket.send(audioBlob);
            socket.send(JSON.stringify({ type: "process_interim_transcript" }));
        }
        utteranceChunks = [];
    };

    mediaRecorder.start();
    updateStatus("Recording your response...", false);
}

function connectWebSocket() {
    const url = BACKEND_WS_URL;
    if (!url || url === "wss://YOUR_RENDER_BACKEND_URL_HERE/ws") {
        updateStatus("Backend URL is not configured in the script.", true);
        updateButtonState('disconnected');
        return;
    }
    if (isConnecting) return;
    isConnecting = true;
    updateStatus(`Connecting to backend...`);
    actionBtn.textContent = 'Connecting...';
    actionBtn.disabled = true;

    if (socket) performFullCleanupSocketOnly();

    let tokenPromise = currentUser ? currentUser.getIdToken(true) : Promise.reject("User not logged in");

    tokenPromise.then(idToken => {
        const wsUrl = `${url}?token=${idToken}`;
        try {
            socket = new WebSocket(wsUrl);
            setupSocketHandlers();
        } catch (e) {
            updateStatus("Error creating WebSocket: " + e.message, true);
            isConnecting = false;
            updateButtonState('disconnected');
        }
    }).catch(error => {
        updateStatus("Authentication error. Cannot connect to backend.", true);
        isConnecting = false;
        updateButtonState(currentUser ? 'disconnected' : 'initial');
    });
}

function setupSocketHandlers() {
    if (!socket) { isConnecting = false; return; }
    fullLogDisplay.innerHTML = "";
    userTranscriptionDisplay.textContent = "";
    setActiveInvestor(null);

    socket.onopen = () => {
        isConnecting = false;
        updateStatus("Connected. Describe your startup to begin.");
        startupDetailsArea.style.display = 'block';
        modeSelectionArea.style.display = 'block';
        pitchingArea.style.display = 'none';
        reportArea.style.display = 'none';
        practiceAgainBtn.style.display = 'none';
        historyArea.style.display = 'none';
        updateButtonState('ready_to_pitch');
    };

    socket.onmessage = async (event) => {
        const messageData = JSON.parse(event.data);
        switch (messageData.type) {
            case 'analysis_report':
                cancelResponseTimer(); setActiveInvestor(null); updateButtonState('report_complete');
                displayAnalysisReport(messageData.data);
                break;
            case 'history_data':
                updateStatus("History loaded."); setActiveInvestor(null);
                displayHistory(messageData.data);
                break;
            case 'session_terminated':
                updateStatus(`Session terminated: ${messageData.reason || ''}`, true);
                cleanupSessionButKeepConnection();
                reportArea.style.display = 'block';
                reportContent.innerHTML = `<h2 style="text-align:center;">Session Ended by Investor</h2><p style="text-align:center;">${messageData.text || ''}</p><p style="text-align:center; font-style:italic;">Reason: ${messageData.reason || 'Not specified'}</p>`;
                practiceAgainBtn.style.display = 'inline-block';
                actionBtn.style.display = 'none';
                stopBtn.style.display = 'none';
                break;
            case 'user_interim_transcript':
                if (messageData.text) {
                    userTranscriptionDisplay.textContent = messageData.text;
                }
                break;
            case 'investor':
                appendToLog(messageData.investor_name, messageData.text);
                if (appState !== 'generating_report') {
                    userHasSpokenInThisTurn = false;
                    startResponseTimer();
                    updateButtonState('listening');
                }
                break;
            case 'user':
                const lastEntry = conversationHistory[conversationHistory.length - 1];
                if(lastEntry?.role === 'User' && lastEntry.content === null) {
                    lastEntry.content = messageData.text;
                }
                break;
            case 'error':
                updateStatus(messageData.text, true);
                if (appState === 'processing') {
                    updateButtonState('listening');
                }
                break;
        }
    };

    socket.onclose = (event) => {
        isConnecting = false;
        if (!['generating_report', 'report_complete', 'history_view', 'initial'].includes(appState)) {
             updateStatus("Disconnected from backend.", true);
        }

        if (currentUser) {
            updateButtonState('disconnected');
            setTimeout(() => {
                if (currentUser && (!socket || socket.readyState === WebSocket.CLOSED)) {
                    connectWebSocket();
                }
            }, 3000);

        } else {
            performFullCleanupAndResetUI();
        }
    };

    socket.onerror = (error) => {
        isConnecting = false;
        updateStatus("WebSocket error. Connection failed.", true);
        updateButtonState(currentUser ? 'disconnected' : 'initial');
    };
}

function performFullCleanupSocketOnly() {
    if (socket) {
        socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null;
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            socket.close();
        }
        socket = null;
    }
}

function cleanupSessionMedia() {
    clearTimeout(pitchTimeout);
    clearInterval(pitchTimerInterval);
    cancelResponseTimer();
    isAudioProcessingInitialized = false;

    if (mediaRecorder?.state === "recording") mediaRecorder.stop();
    mediaRecorder = null;
    if (masterRecorder?.state === "recording") masterRecorder.stop();
    masterRecorder = null;

    if (processorNode) {
        processorNode.disconnect();
        processorNode = null;
    }
    if (mediaStreamSource) {
        mediaStreamSource.disconnect();
        mediaStreamSource = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().catch(e => {});
    }
    audioContext = null;
}

function cleanupSessionButKeepConnection() {
    cleanupSessionMedia();
    pitchingArea.style.display = 'none';
    timerDiv.style.display = 'none';
    setActiveInvestor(null);
    userTranscriptionDisplay.textContent = "";
    responseControls.style.display = 'none';
    finishedSpeakingBtn.style.display = 'none';
}

function performFullCleanupAndResetUI() {
    cleanupSessionMedia();
    performFullCleanupSocketOnly();
    pitchingArea.style.display = 'none';
    reportArea.style.display = 'none';
    practiceAgainBtn.style.display = 'none';
    historyArea.style.display = 'none';
    timerDiv.style.display = 'none';
    modeSelectionArea.style.display = 'none';
    startupDetailsArea.style.display = 'none';
    updateButtonState('initial');
    setActiveInvestor(null);
    fullLogDisplay.innerHTML = "";
    userTranscriptionDisplay.textContent = "";
    responseControls.style.display = 'none';
    finishedSpeakingBtn.style.display = 'none';
}

function appendToLog(role, text) {
    const entry = document.createElement('div');
    entry.classList.add('transcription-entry');

    if (role === "You") {
        setActiveInvestor(null);
        const utteranceStartTime = (Date.now() - pitchStartTime) / 1000;
        conversationHistory.push({ role: 'User', content: text, startTime: utteranceStartTime });

        const prefix = document.createElement('span');
        prefix.style.fontWeight = 'bold';
        prefix.textContent = "You: ";
        entry.appendChild(prefix);
        entry.classList.add('user-text-log');
    } else {
        const currentTime = (Date.now() - pitchStartTime) / 1000;
        conversationHistory.push({ role: `Investor (${role})`, content: text, startTime: currentTime });
        setActiveInvestor(role);
        if (investorSpeechBubbles[role]) {
            investorSpeechBubbles[role].textContent = text;
        }
        investorExpectedToRespond = role;
        const nameSpan = document.createElement('span');
        nameSpan.classList.add('investor-name-log', `investor-${role.toLowerCase().replace(/\s+/g, '-')}`);
        nameSpan.textContent = `${role}: `;
        entry.appendChild(nameSpan);
    }
    entry.appendChild(document.createTextNode(text));
    fullLogDisplay.appendChild(entry);
    fullLogDisplay.scrollTop = fullLogDisplay.scrollHeight;
}


function displayAnalysisReport(data) {
    pitchingArea.style.display = 'none';
    historyArea.style.display = 'none';
    reportArea.style.display = 'block';
    reportContent.innerHTML = '';
    practiceAgainBtn.style.display = 'inline-block';

    const title = document.createElement('h1');
    title.textContent = 'Pitch Analysis Report';
    reportContent.appendChild(title);

    const createSection = (titleText, contentEl) => {
        const section = document.createElement('div');
        section.className = 'report-section';
        const h2 = document.createElement('h2');
        h2.textContent = titleText;
        section.appendChild(h2);
        if (contentEl) section.appendChild(contentEl);
        reportContent.appendChild(section);
    };
    const createList = (itemsArray) => {
        const ul = document.createElement('ul');
        ul.className = 'report-list';
        const items = Array.isArray(itemsArray) ? itemsArray : (itemsArray ? [itemsArray] : []);
        if (items.length === 0 || (items.length === 1 && !items[0])) {
            const li = document.createElement('li');
            li.textContent = 'No specific feedback points available.';
            li.style.fontStyle = 'italic';
            ul.appendChild(li);
        } else {
            items.forEach(item => {
                if (item) {
                    const li = document.createElement('li');
                    li.textContent = String(item).replace(/^- |^\d\. /, '');
                    ul.appendChild(li);
                }
            });
        }
        return ul;
    };

    const analysisReportFromData = data.analysis_report || {};
    const sessionMode = data.mode;

    if (sessionMode === "strict") {
        const assessment = document.createElement('p');
        assessment.className = 'status-assessment';
        assessment.textContent = analysisReportFromData.default_alive_dead || 'Assessment not available.';
        if (String(assessment.textContent).toLowerCase().includes('alive')) assessment.classList.add('alive');
        else if (String(assessment.textContent).toLowerCase().includes('dead')) assessment.classList.add('dead');
        else assessment.classList.add('neutral');
        createSection('Default Alive / Default Dead', assessment);

        const pillarsEl = document.createElement('div');
        if (analysisReportFromData.pillars && Object.keys(analysisReportFromData.pillars).length > 0) {
            for (const pillarName in analysisReportFromData.pillars) {
                const pillarData = analysisReportFromData.pillars[pillarName];
                const h3 = document.createElement('h3');
                h3.textContent = pillarName;
                const scoreSpan = document.createElement('span');
                scoreSpan.className = 'score';
                scoreSpan.textContent = pillarData.score || 'N/A';
                h3.appendChild(scoreSpan);
                pillarsEl.appendChild(h3);
                pillarsEl.appendChild(createList(pillarData.feedback));
            }
            createSection('Core Pillar Analysis', pillarsEl);
        } else {
            createSection('Core Pillar Analysis', createList(["Pillar analysis not available."]));
        }
        createSection('Brutally Honest Feedback', createList(analysisReportFromData.brutal_feedback));
        createSection('Top 3 Areas for Next Practice', createList(analysisReportFromData.top_3_areas));
    } else {
        const summaryEl = document.createElement('p');
        summaryEl.textContent = analysisReportFromData.message || "Session complete.";
        createSection("Session Summary", summaryEl);

        const transcriptForDrillOrFallback = analysisReportFromData.transcript || data.replay_data?.transcript || [];
        if (transcriptForDrillOrFallback.length > 0) {
            const transcriptContainer = document.createElement('div');
            transcriptContainer.style.maxHeight = '400px'; transcriptContainer.style.overflowY = 'auto'; transcriptContainer.style.border = '1px solid #eee'; transcriptContainer.style.padding = '10px'; transcriptContainer.style.marginTop = '15px'; transcriptContainer.style.backgroundColor = '#f8f9fa';
            transcriptForDrillOrFallback.forEach((item) => {
                const p = document.createElement('p');
                p.style.marginBottom = '8px'; p.style.padding = '6px'; p.style.borderRadius = '4px';
                if (item.role === 'User') p.style.backgroundColor = 'var(--mdc-theme-background)';
                else if (item.role && item.role.startsWith('Investor')) p.style.backgroundColor = '#e8f0fe';
                p.innerHTML = `<strong>${item.role || 'System'}:</strong> ${item.content || item.text || "<i>...</i>"}`;
                transcriptContainer.appendChild(p);
            });
            createSection("Session Transcript", transcriptContainer);
        } else {
             createSection("Session Transcript", createList(["Full transcript for this session should be in the replay if audio was captured."]));
        }
    }

    const currentReplayData = data.replay_data;
    if (currentReplayData?.audio_path) {
        const replayContainerHTML = `<div id="replayContainer" style="margin-top: 25px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;"><h2 style="text-align: center;">Replay Simulation</h2><audio id="replayAudio" controls style="width: 100%; margin-bottom: 10px;"></audio><div id="replayTranscript" style="border-top: 1px solid #eee; margin-top: 15px; max-height: 300px; overflow-y: auto; padding: 10px; background-color: #f8f9fa; border-radius: 6px;"></div></div>`;
        reportContent.insertAdjacentHTML('beforeend', replayContainerHTML);
        const replayAudio = document.getElementById('replayAudio');
        const replayTranscriptEl = document.getElementById('replayTranscript');

        let httpUrlForReplay = BACKEND_WS_URL.replace('wss://', 'https://').replace('/ws', '');
        if (httpUrlForReplay.endsWith('/')) httpUrlForReplay = httpUrlForReplay.slice(0,-1);
        replayAudio.src = httpUrlForReplay + currentReplayData.audio_path;

        const transcriptToDisplayForReplay = currentReplayData.transcript || [];
        if (transcriptToDisplayForReplay && Array.isArray(transcriptToDisplayForReplay)) {
            replayTranscriptEl.innerHTML = '';
            transcriptToDisplayForReplay.forEach((item, index) => {
                const p = document.createElement('p');
                p.id = `transcript-line-${index}`;
                p.innerHTML = `<strong>${item.role || 'System'}:</strong> ${item.content || item.text || "<i>...speaking...</i>"}`;
                replayTranscriptEl.appendChild(p);
            });
            let lastHighlightedIndex = -1;
            replayAudio.ontimeupdate = () => {
                const currentTime = replayAudio.currentTime;
                let currentHighlightIndex = -1;
                for(let i=0; i < transcriptToDisplayForReplay.length; i++) {
                    const item = transcriptToDisplayForReplay[i]; const nextItem = transcriptToDisplayForReplay[i+1];
                    const isUserActive = item.role === 'User' && typeof item.startTime === 'number' && typeof item.endTime === 'number' && currentTime >= item.startTime && currentTime < item.endTime;
                    const isInvestorActive = (item.role && !item.role.startsWith('User') && item.role !== 'System') && typeof item.startTime === 'number' && currentTime >= item.startTime && (!nextItem || typeof nextItem.startTime !== 'number' || currentTime < nextItem.startTime);
                    if (isUserActive || isInvestorActive) { currentHighlightIndex = i; break; }
                }
                if (currentHighlightIndex !== lastHighlightedIndex) {
                    if(lastHighlightedIndex !== -1 && document.getElementById(`transcript-line-${lastHighlightedIndex}`)) document.getElementById(`transcript-line-${lastHighlightedIndex}`).classList.remove('highlight');
                    if(currentHighlightIndex !== -1) {
                        const lineEl = document.getElementById(`transcript-line-${currentHighlightIndex}`);
                        if(lineEl) { lineEl.classList.add('highlight'); lineEl.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' }); }
                    }
                    lastHighlightedIndex = currentHighlightIndex;
                }
            };
        } else {
            replayTranscriptEl.innerHTML = "<p>Transcript not available for replay.</p>";
        }
    }
}


function handleViewHistoryClick() {
    if (!currentUser) { alert("Please log in to view history."); return; }
    const userIdentifierForHistory = currentUser.email;
    if (socket?.readyState === WebSocket.OPEN) {
        updateStatus("Fetching session history...");
        startupDetailsArea.style.display = 'none'; modeSelectionArea.style.display = 'none'; pitchingArea.style.display = 'none'; reportArea.style.display = 'none';
        practiceAgainBtn.style.display = 'none';
        historyArea.style.display = 'block';
        updateButtonState('history_view');
        socket.send(JSON.stringify({ type: "get_history", identifier: userIdentifierForHistory }));
    } else {
        alert("Please connect to the backend first. If connection fails, check configuration.");
    }
}

function displayHistory(historyData) {
    startupDetailsArea.style.display = 'none'; pitchingArea.style.display = 'none'; reportArea.style.display = 'none'; historyArea.style.display = 'block';
    sessionList.innerHTML = '';
    const noGraphMsgElement = document.querySelector('p.no-graph-data-msg');
    if(noGraphMsgElement) noGraphMsgElement.remove();

    if (!historyData || historyData.length === 0) {
        sessionList.innerHTML = '<p>No history found.</p>';
        if (progressChart) progressChart.destroy();
        progressChart = null;
        progressChartCanvas.style.display = 'none';
        return;
    }
    historyData.sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0)).reverse();

    const graphDataSlice = historyData.slice(0, 30).reverse();

    const labels = graphDataSlice.map(item => new Date(item.timestamp).toLocaleDateString());
    const scores = {'Problem/Solution Fit': [], 'Formidable Founders (Clarity & Conviction)': [], 'Market / \'Why Now?\'': []};
    let hasStrictDataForGraph = false;

    graphDataSlice.forEach(item => {
        const analysisReportForScores = item.analysis_report || {};
        if (item.mode === "strict" && analysisReportForScores && analysisReportForScores.pillars) {
            hasStrictDataForGraph = true;
            for (const pillarName in scores) {
                const scoreValue = analysisReportForScores.pillars[pillarName]?.score;
                scores[pillarName].push(typeof scoreValue === 'number' ? scoreValue : null);
            }
        } else {
            for (const pillarName in scores) { scores[pillarName].push(null); }
        }
    });

    if (progressChart) progressChart.destroy();
    if (hasStrictDataForGraph) {
        progressChartCanvas.style.display = 'block';
        progressChart = new Chart(progressChartCanvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Problem/Solution Fit', data: scores['Problem/Solution Fit'], borderColor: 'rgba(26, 115, 232, 1)', tension: 0.1, spanGaps: true, backgroundColor: 'rgba(26, 115, 232, 0.2)' },
                    { label: 'Formidable Founders', data: scores['Formidable Founders (Clarity & Conviction)'], borderColor: 'rgba(52, 168, 83, 1)', tension: 0.1, spanGaps: true, backgroundColor: 'rgba(52, 168, 83, 0.2)' },
                    { label: 'Market / Why Now?', data: scores['Market / \'Why Now?\''], borderColor: 'rgba(234, 67, 53, 1)', tension: 0.1, spanGaps: true, backgroundColor: 'rgba(234, 67, 53, 0.2)' }
                ]
            },
            options: { scales: { y: { beginAtZero: true, max: 5, ticks: { stepSize: 1 } } }, responsive: true, maintainAspectRatio: true }
        });
    } else {
        progressChartCanvas.style.display = 'none';
        if (historyData.some(item => item.mode === "strict")) {
             const existingMsg = document.querySelector('p.no-graph-data-msg');
             if (!existingMsg && sessionList.parentNode) {
                const noGraphMsg = document.createElement('p');
                noGraphMsg.textContent = 'No scorable data in the recent strict mode sessions to display graph.';
                noGraphMsg.className = 'no-graph-data-msg'; noGraphMsg.style.textAlign = 'center'; noGraphMsg.style.fontStyle = 'italic'; noGraphMsg.style.color = '#777';
                sessionList.parentNode.insertBefore(noGraphMsg, progressChartCanvas.nextSibling);
             }
        }
    }

    historyData.forEach(item => {
        const entryDiv = document.createElement('div');
        entryDiv.className = 'history-entry';
        const summary = document.createElement('div');
        summary.className = 'history-summary';
        summary.innerHTML = `<span>${new Date(item.timestamp).toLocaleString()} (${item.mode || 'N/A'})</span> <span>Click to view report</span>`;
        entryDiv.appendChild(summary);
        entryDiv.onclick = () => {
            historyArea.style.display = 'none';
            displayAnalysisReport(item);
            updateButtonState('report_complete');
        };
        sessionList.appendChild(entryDiv);
    });
}

function startPitchTimer() {
    let timeLeft = PITCH_DURATION_MS;
    timerDiv.style.display = 'block';
    const updateTimerDisplay = () => {
        const minutes = Math.floor((timeLeft / 1000) / 60);
        const seconds = Math.floor((timeLeft / 1000) % 60);
        timerDiv.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        timeLeft -= 1000;
    };
    updateTimerDisplay();
    pitchTimerInterval = setInterval(updateTimerDisplay, 1000);
    pitchTimeout = setTimeout(endPitchSession, PITCH_DURATION_MS);
}

function cancelResponseTimer() { clearTimeout(responseTimeout); responseTimeout = null; }
function startResponseTimer() { cancelResponseTimer(); responseTimeout = setTimeout(handleResponseTimeout, RESPONSE_TIMEOUT_MS); }

function handleResponseTimeout() {
    cancelResponseTimer(); // Clear the timeout as it has fired
    if (socket?.readyState === WebSocket.OPEN) {
        const currentTime = (Date.now() - pitchStartTime) / 1000;
        // Add a system message to history indicating timeout
        conversationHistory.push({role: 'System', content: '[System: Founder showed hesitation and failed to respond in time.]', startTime: currentTime});
        const userIdentifierForSession = currentUser ? currentUser.email : "local_user";
        socket.send(JSON.stringify({ type: "user_timeout", identifier: userIdentifierForSession })); // Notify backend of timeout
        
        // Do NOT call updateButtonState('listening') here, as that clears the transcription.
        // Instead, directly set the status and ensure the confirm/reset buttons are visible.
        updateStatus("You took too long to respond. Please review your response and confirm, or reset to try again.", true);
        responseControls.style.display = 'flex'; // Ensure confirm/reset buttons are visible
        finishedSpeakingBtn.style.display = 'none'; // Ensure finished speaking button is hidden
        appState = 'awaiting_confirmation'; // Set appState to allow confirm/reset

        if (investorExpectedToRespond && investorAvatars[investorExpectedToRespond]) {
             investorAvatars[investorExpectedToRespond].classList.add('thinking'); // Keep investor thinking state
        }
    }
}

function updateStatus(message, isError = false) {
    statusDiv.textContent = `Status: ${message}`;
    statusDiv.className = isError ? 'error' : '';
}

function setActiveInvestor(investorName) {
    Object.values(investorAvatars).forEach(avatar => avatar.classList.remove('speaking', 'thinking'));
    Object.values(investorSpeechBubbles).forEach(bubble => bubble.classList.remove('visible'));

    if (investorName) {
        if (investorAvatars[investorName]) {
            investorAvatars[investorName].classList.add('speaking');
        }
        if (investorSpeechBubbles[investorName]) {
            investorSpeechBubbles[investorName].classList.add('visible');
        }
        container.classList.add('speech-active');
        currentSpeakingInvestor = investorName;
    } else {
        container.classList.remove('speech-active');
        currentSpeakingInvestor = null;
    }
}

function populateProfileDropdown() {
    startupProfileSelect.innerHTML = '<option value="">-- Select or Enter Details Below --</option>';
    savedStartupProfiles.forEach(profile => {
        const option = document.createElement('option');
        option.value = profile.id; option.textContent = profile.name;
        startupProfileSelect.appendChild(option);
    });
}

function clearMainStartupInputs() {
    mainStartupNameInput.value = ""; mainOneLinerInput.value = ""; mainCoreProblemInput.value = "";
    currentSelectedProfileData = null;
    startupProfileSelect.value = "";
}

async function loadStartupProfiles() {
    if (!currentUser) return;
    try {
        const snapshot = await fbDb.collection('users').doc(currentUser.uid).collection('startup_profiles').orderBy('lastUsedAt', 'desc').get();
        savedStartupProfiles = [];
        snapshot.forEach(doc => { savedStartupProfiles.push({ id: doc.id, ...doc.data() }); });
        populateProfileDropdown();
        if (savedStartupProfiles.length > 0) {
           startupProfileSelect.value = savedStartupProfiles[0].id;
           handleProfileSelectionChange();
        } else { clearMainStartupInputs(); }
    } catch (error) { console.error("Error loading startup profiles:", error); }
}

function handleProfileSelectionChange() {
    const selectedId = startupProfileSelect.value;
    if (selectedId) {
        currentSelectedProfileData = savedStartupProfiles.find(p => p.id === selectedId);
        if (currentSelectedProfileData) {
            mainStartupNameInput.value = currentSelectedProfileData.name;
            mainOneLinerInput.value = currentSelectedProfileData.oneLiner;
            mainCoreProblemInput.value = currentSelectedProfileData.problem;
        }
    } else { clearMainStartupInputs(); }
}

async function openManageProfilesModal() {
    if (!currentUser) return;
    profileListContainer.innerHTML = 'Loading profiles...';
    profilesModal.style.display = "block";
    profileFormContainer.style.display = "none";
    try {
        const snapshot = await fbDb.collection('users').doc(currentUser.uid).collection('startup_profiles').orderBy('name').get();
        savedStartupProfiles = [];
        snapshot.forEach(doc => { savedStartupProfiles.push({ id: doc.id, ...doc.data() }); });
        renderProfileListInModal();
        populateProfileDropdown();
    } catch (error) { console.error("Error fetching profiles for modal:", error); profileListContainer.innerHTML = 'Error loading profiles.'; }
}

function renderProfileListInModal() {
    profileListContainer.innerHTML = '';
    if (savedStartupProfiles.length === 0) {
        profileListContainer.innerHTML = '<p>No saved profiles yet. Click "Create New Profile" to add one.</p>';
        return;
    }
    savedStartupProfiles.forEach(profile => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'profile-item';
        itemDiv.innerHTML = `<span>${profile.name} <em style="font-size:0.8em; color:#777;">(${(profile.oneLiner || "").substring(0,30)}...)</em></span><div class="profile-actions"><button class="btn edit-profile-btn" data-id="${profile.id}" style="background-color:#ffe082; color:#212529; padding: 5px 10px; font-size:0.8rem; box-shadow: none;">Edit</button><button class="btn delete-profile-btn" data-id="${profile.id}" style="background-color:#ef9a9a; padding: 5px 10px; font-size:0.8rem; box-shadow: none;">Delete</button></div>`;
        profileListContainer.appendChild(itemDiv);
    });

    document.querySelectorAll('.edit-profile-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const profileId = e.target.dataset.id;
            const profileToEdit = savedStartupProfiles.find(p => p.id === profileId);
            if (profileToEdit) {
                profileFormTitle.textContent = "Edit Profile"; editingProfileIdInput.value = profileId;
                modalStartupNameInput.value = profileToEdit.name; modalOneLinerInput.value = profileToEdit.oneLiner; modalCoreProblemInput.value = profileToEdit.problem;
                profileFormContainer.style.display = "block";
            }
        });
    });
    document.querySelectorAll('.delete-profile-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            const profileId = e.target.dataset.id;
            if (confirm("Are you sure you want to delete this profile?")) { await deleteStartupProfile(profileId); }
        });
    });
}

async function saveStartupProfile() {
    if (!currentUser) return;
    const profileName = modalStartupNameInput.value.trim(); const oneLiner = modalOneLinerInput.value.trim(); const problem = modalCoreProblemInput.value.trim(); const profileId = editingProfileIdInput.value;
    if (!profileName || !oneLiner || !problem) { alert("All fields are required for the startup profile."); return; }
    const profileData = { name: profileName, oneLiner: oneLiner, problem: problem, lastUsedAt: firebase.firestore.FieldValue.serverTimestamp() };
    try {
        let updatedProfileId = profileId;
        if (profileId) {
            await fbDb.collection('users').doc(currentUser.uid).collection('startup_profiles').doc(profileId).update(profileData);
            alert("Profile updated successfully!");
        } else {
            profileData.createdAt = firebase.firestore.FieldValue.serverTimestamp();
            const docRef = await fbDb.collection('users').doc(currentUser.uid).collection('startup_profiles').add(profileData);
            updatedProfileId = docRef.id;
            alert("Profile created successfully!");
        }
        profileFormContainer.style.display = "none";
        await loadStartupProfiles();
        renderProfileListInModal();
        if(updatedProfileId) startupProfileSelect.value = updatedProfileId;
        handleProfileSelectionChange();
    } catch (error) { console.error("Error saving profile:", error); alert("Error saving profile: " + error.message); }
}

async function deleteStartupProfile(profileId) {
    if (!currentUser || !profileId) return;
    try {
        await fbDb.collection('users').doc(currentUser.uid).collection('startup_profiles').doc(profileId).delete();
        alert("Profile deleted successfully.");
        if (startupProfileSelect.value === profileId) clearMainStartupInputs();
        await loadStartupProfiles();
        renderProfileListInModal();
    } catch (error) { console.error("Error deleting profile:", error); alert("Error deleting profile: " + error.message); }
}

updateButtonState('initial');
