// @ts-nocheck




import AsyncStorage from '@react-native-async-storage/async-storage';

import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';



const testPDF = async () => {
  const html = `<h1>Hello PDF</h1><p>PDF works</p>`;
  const { uri } = await Print.printToFileAsync({ html });
  await Sharing.shareAsync(uri);
};


import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import { Audio } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system/legacy';




 // Use legacy API for compatibility
import { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

import { Vibration } from "react-native";



// ✅ Use your computer's IP address
const API_BASE_URL = "http://172.20.127.151:8000";



// =====================
// 📦 Persistent Runtime Logger
// =====================
const LOG_FILE = FileSystem.documentDirectory + "runtime_logs.json";

const persistLog = async (
  level: "INFO" | "WARN" | "ERROR",
  message: string,
  payload?: any
) => {
  try {
    const existing = await FileSystem.readAsStringAsync(LOG_FILE).catch(() => "[]");
    const logs = JSON.parse(existing);

    logs.push({
      timestamp: new Date().toISOString(),
      level,
      message,
      payload,
    });

    await FileSystem.writeAsStringAsync(
      LOG_FILE,
      JSON.stringify(logs, null, 2)
    );
  } catch (e) {
    console.warn("Failed to persist log");
  }
};

// Function to convert base64 to image source


const base64ToImageSource = (base64String: string) => {
  return { uri: `data:image/png;base64,${base64String}` };
};

export default function ExploreScreen() {
  const [selectedFile, setSelectedFile] = useState<DocumentPicker.DocumentPickerResult | null>(null);
  const [results, setResults] = useState<any>(null);
  const [visualizations, setVisualizations] = useState<{[key: string]: string} | null>(null);
  const [loading, setLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'unknown' | 'connected' | 'failed'>('unknown');
  

    // 🌙 Night Preferences (ADDED – non-intrusive)
 const [nightPreferences, setNightPreferences] = useState({
  quiet_start: "23:00",
  quiet_end: "07:00",
  sensitivity: "medium",
  max_nudges_per_hour: 2,
  cooldown_minutes: 15,
  nudge_type: "vibration",
});


  // Audio recording states
  const [isRecording, setIsRecording] = useState(false);
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [recordingTimer, setRecordingTimer] = useState<NodeJS.Timeout | null>(null);
  const [recordedUri, setRecordedUri] = useState<string | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);

  // Request microphone permissions
  useEffect(() => {
    const requestPermission = async () => {
      try {
        const { status } = await Audio.requestPermissionsAsync();
        console.log('Microphone permission status:', status);
        setHasPermission(status === 'granted');
      } catch (error) {
        console.error('Failed to get permission:', error);
        setHasPermission(false);
      }
    };

    requestPermission();
  }, []);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (recordingTimer) {
        clearInterval(recordingTimer);
        setRecordingTimer(null);
      }
    };
  }, [recordingTimer]);

  const startRecording = async () => {
    try {
      if (hasPermission === false) {
        Alert.alert(
          'Permission Required', 
          'Please grant microphone permission in your device settings to record audio',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'OK', style: 'default' }
          ]
        );
        return;
      }

      // Configure audio recording
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      console.log('Creating new recording...');
      
      // Create new recording
      const newRecording = new Audio.Recording();
      
      try {
        // Use a simpler configuration that works on both iOS and Android
        await newRecording.prepareToRecordAsync(
          Audio.RecordingOptionsPresets.HIGH_QUALITY
        );

        await newRecording.startAsync();
        console.log('Recording started successfully');

        await persistLog("INFO", "Recording started");

        
        setRecording(newRecording);
        setIsRecording(true);
        setRecordingDuration(0);
        setRecordedUri(null);
        setSelectedFile(null);
        setResults(null);
        setVisualizations(null);

        // Start timer
        const timer = setInterval(() => {
          setRecordingDuration(prev => prev + 1);
        }, 1000);
        setRecordingTimer(timer);

        Alert.alert('Recording Started', 'Recording audio for snoring analysis...');

      } catch (prepareError) {
        console.error('Failed to prepare recording:', prepareError);
        Alert.alert('Recording Error', 'Failed to prepare audio recording. Please try again.');
      }

    } catch (error) {
      console.error('Failed to start recording:', error);
      Alert.alert('Recording Error', 'Failed to start recording. Please check microphone permissions.');
    }
  };

  const stopRecording = async () => {
    try {
      if (!recording) return;

      console.log('Stopping recording...');
      
      // Stop timer first
      if (recordingTimer) {
        clearInterval(recordingTimer);
        setRecordingTimer(null);
      }

      // Stop the recording
      await recording.stopAndUnloadAsync();
      
      // Get the recording URI
      const uri = recording.getURI();
      console.log('Recording URI:', uri);

      await persistLog("INFO", "Recording stopped", { uri });

      
      if (uri) {
        setRecordedUri(uri);
        
        // Try to get file info with error handling
        let fileSize = 0;
        try {
          // Using the legacy API to avoid deprecation warnings
          const fileInfo = await FileSystem.getInfoAsync(uri);
          console.log('File info:', fileInfo);
          fileSize = fileInfo.size || 0;
        } catch (fileError) {
          console.warn('Could not get file info:', fileError);
          // Continue without file size
        }
        
        // Create a file object similar to DocumentPicker result
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `snoring_recording_${timestamp}.m4a`;
        
        const fileResult: DocumentPicker.DocumentPickerResult = {
          canceled: false,
          assets: [{
            uri,
            name: filename,
            mimeType: 'audio/m4a',
            size: fileSize,
          }]
        };
        setSelectedFile(fileResult);
      }

      setIsRecording(false);
      setRecording(null);

      Alert.alert('Recording Complete', `Audio recorded for ${recordingDuration} seconds`);

    } catch (error) {
      console.error('Failed to stop recording:', error);
      Alert.alert('Recording Error', 'Failed to stop recording');
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const pickAudioFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'audio/*',
        copyToCacheDirectory: true,
      });

      if (result.assets && result.assets.length > 0) {
        setSelectedFile(result);
        setRecordedUri(null);
        setResults(null);
        setVisualizations(null);
        console.log('Selected file:', result.assets[0].name);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick file');
      console.error('File pick error:', error);
    }
  };

  const testConnection = async () => {
    try {
      console.log('Testing connection to:', `${API_BASE_URL}/health`);
      setConnectionStatus('unknown');
      
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000,
      });
      
      console.log('✅ Connection test successful:', response.data);
      setConnectionStatus('connected');
      Alert.alert('Success', 'Connected to server successfully!');
      return true;
    } catch (error: any) {
      console.error('❌ Connection test failed:', error);
      setConnectionStatus('failed');
      
      let errorMessage = 'Cannot connect to server. Please check:\n\n';
      
      if (error.code === 'ECONNREFUSED') {
        errorMessage += '• Backend is not running\n• Port 8000 is blocked\n• Wrong IP address';
      } else if (error.message?.includes('Network Error')) {
        errorMessage += '• Devices not on same WiFi\n• Firewall blocking connection\n• Wrong IP address';
      } else {
        errorMessage += error.message || 'Unknown error occurred';
      }
      
      Alert.alert('Connection Failed', errorMessage);
      return false;
    }
  };

    // 🌙 Save night preferences before analysis (ADDED)
  const saveNightPreferences = async () => {
    try {
      await axios.post(`${API_BASE_URL}/night-preferences`, nightPreferences);
      console.log("Night preferences saved");
    } catch (err) {
      console.warn("Night preferences save failed (non-blocking)", err);
    }
  };

  const saveSessionLocally = async (analysisResult: any) => {
  try {
    const existing = await AsyncStorage.getItem("night_sessions");
    const sessions = existing ? JSON.parse(existing) : [];

        // Quiet Snore Minutes calculation
    const snoringIntervals = analysisResult.analysis.snoring_intervals || [];

    const totalSnoringDuration = snoringIntervals.reduce(
      (sum: number, interval: any) => sum + (interval.duration || 0),
      0
    );

    const quietMinutes =
      Math.max(analysisResult.audio_duration - totalSnoringDuration, 0);


    const session = {
      id: Date.now().toString(),
      startTime: new Date().toISOString(),
      audioDuration: analysisResult.audio_duration,
      snoringRatio: analysisResult.analysis.snoring_ratio,
      totalSegments: analysisResult.analysis.total_segments,
      snoringSegments: analysisResult.analysis.snoring_segments,
      snoringIntervals: analysisResult.analysis.snoring_intervals,
            quietMinutes,
      quietMinutes: quietMinutes,

      nudges: analysisResult.nudges,
      preferences: nightPreferences,
    };

    sessions.push(session);

    await AsyncStorage.setItem(
      "night_sessions",
      JSON.stringify(sessions)
    );

    await persistLog("INFO", "Session stored locally", session);

  } catch (err) {
    console.warn("❌ Failed to save session", err);
  }
};

const debugReadSessions = async () => {
  const data = await AsyncStorage.getItem("night_sessions");
  console.log("📦 Stored sessions:", JSON.parse(data || "[]"));
};

const debugReadLogs = async () => {
  try {
    const data = await FileSystem.readAsStringAsync(LOG_FILE);
    console.log("🧾 Persisted runtime logs:", JSON.parse(data));
  } catch (e) {
    console.warn("No runtime logs found yet");
  }
};



const exportSessionsAsCSV = async () => {
  try {
    const data = await AsyncStorage.getItem("night_sessions");
    if (!data) {
      Alert.alert("No Data", "No sessions available to export");
      return;
    }

    const sessions = JSON.parse(data);

    const header = [
      "Session ID",
      "Start Time",
      "Audio Duration",
      "Snoring Ratio",
      "Total Segments",
      "Snoring Segments",
      "Nudges Sent"
    ];

    const rows = sessions.map((s: any) => [
      s.id,
      s.startTime,
      s.audioDuration,
      s.snoringRatio,
      s.totalSegments,
      s.snoringSegments,
      s.nudges?.nudges_this_hour ?? 0
    ]);

    const csv = [header, ...rows]
      .map(row => row.join(","))
      .join("\n");

    const fileUri =
      FileSystem.documentDirectory + "snoring_sessions.csv";

    await FileSystem.writeAsStringAsync(fileUri, csv);

    await Sharing.shareAsync(fileUri);

    Alert.alert("CSV Exported", `Saved to:\n${fileUri}`);
  } catch (err) {
    console.error(err);
    Alert.alert("Export Failed", "Could not export CSV");
  }
};

const exportSessionsAsPDF = async () => {
  const data = await AsyncStorage.getItem("night_sessions");
  if (!data) {
    Alert.alert("No Data", "No sessions available");
    return;
  }

  const sessions = JSON.parse(data);

  const rows = sessions.map((s: any, i: number) => `
    <tr>
      <td>${i + 1}</td>
      <td>${new Date(s.startTime).toLocaleString()}</td>
      <td>${s.audioDuration}s</td>
      <td>${(s.snoringRatio * 100).toFixed(1)}%</td>
      <td>${s.snoringSegments}/${s.totalSegments}</td>
    </tr>
  `).join("");

  const html = `
    <html>
      <body style="font-family: Arial; padding: 16px;">
        <h1>Snoring Summary Report</h1>
        <p>Total Sessions: ${sessions.length}</p>

        <table border="1" cellspacing="0" cellpadding="8" width="100%">
          <tr>
            <th>#</th>
            <th>Date</th>
            <th>Duration</th>
            <th>Snoring %</th>
            <th>Segments</th>
          </tr>
          ${rows}
        </table>
      </body>
    </html>
  `;

  const { uri } = await Print.printToFileAsync({ html });
  await Sharing.shareAsync(uri);
};

// 🔔 Single gentle-long vibration nudge
const triggerNudgeVibration = () => {
  // 600ms single vibration (noticeable but not alarming)
  Vibration.vibrate(600);
};




  const uploadAudio = async () => {
    if (!selectedFile?.assets?.[0]) {
      Alert.alert('Error', 'Please record or select an audio file first');
      return;
    }

    setLoading(true);
    setResults(null);
    setVisualizations(null);

    try {

       await saveNightPreferences();
      // Test connection first
      console.log('Testing connection before upload...');
      const isConnected = await testConnection();
      if (!isConnected) {
        throw new Error('Cannot connect to server. Please check connection and try again.');
      }

      const formData = new FormData();
      const file = selectedFile.assets[0];
      
      // Determine file type based on extension
      let mimeType = 'audio/mpeg'; // Default to MP3
      const fileName = file.name.toLowerCase();
      if (fileName.endsWith('.m4a') || fileName.endsWith('.mp4')) {
        mimeType = 'audio/mp4';
      } else if (fileName.endsWith('.wav')) {
        mimeType = 'audio/wav';
      } else if (fileName.endsWith('.m4a')) {
        mimeType = 'audio/m4a';
      }
      
      // @ts-ignore - Expo types issue with FormData
      formData.append('file', {
        uri: file.uri,
        type: mimeType,
        name: file.name,
      });

      console.log('Starting upload to:', `${API_BASE_URL}/analyze-snoring`);
      console.log('File:', file.name, 'Type:', mimeType);

      const response = await axios.post(`${API_BASE_URL}/analyze-snoring`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes timeout
      });

      console.log('✅ Upload successful, response received');

      await persistLog("INFO", "Upload successful", {
  file: file.name,
  duration: recordingDuration,
});

      setResults(response.data);

      // 🔔 Trigger vibration if backend decided a nudge is needed
if (response.data?.nudges?.nudges_this_hour > 0) {
  triggerNudgeVibration();
}


      await saveSessionLocally(response.data);

      await debugReadSessions();

      await debugReadLogs();



      
      // Set visualizations if available
      if (response.data.visualizations) {
        setVisualizations(response.data.visualizations);
      }

    } catch (error: any) {
      console.error('❌ Upload failed:', error);

      await persistLog("ERROR", "Upload failed", error?.message);

      
      let errorMessage = 'Upload failed. ';
      
      if (error.code === 'ECONNREFUSED') {
        errorMessage = `Cannot connect to server at ${API_BASE_URL}. Please check:\n\n1. Backend is running\n2. Both devices on same WiFi\n3. Firewall allows Python\n4. Correct IP address: 192.168.0.101`;
      } else if (error.response) {
        errorMessage = `Server error: ${error.response.data?.detail || 'Unknown server error'}`;
      } else if (error.request) {
        errorMessage = 'No response from server. Check your connection.';
      } else if (error.message) {
        errorMessage = error.message;
      }

      Alert.alert('Upload Failed', errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#4CAF50';
      case 'failed': return '#F44336';
      default: return '#FF9800';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected to Server';
      case 'failed': return 'Connection Failed';
      default: return 'Not Tested';
    }
  };

  const getSnoringLevel = (ratio: number) => {
    if (ratio > 0.3) return { text: 'HIGH', color: '#ff4444', icon: 'warning' };
    if (ratio > 0.1) return { text: 'MODERATE', color: '#ffaa00', icon: 'alert' };
    return { text: 'LOW', color: '#00aa00', icon: 'checkmark' };
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="mic" size={32} color="#007AFF" />
        <Text style={styles.title}>Snoring Detection</Text>
        <Text style={styles.subtitle}>Record audio or upload audio file to analyze snoring patterns</Text>
        
        
        
        
        {/* Connection Info */}
        <View style={styles.connectionSection}>
          <View style={styles.connectionRow}>
            <Ionicons name="server" size={16} color="#666" />
            <Text style={styles.connectionText}>Server: {API_BASE_URL}</Text>
          </View>
          
          <View style={styles.connectionRow}>
            <View 
              style={[
                styles.statusIndicator, 
                { backgroundColor: getConnectionStatusColor() }
              ]} 
            />
            <Text style={styles.statusText}>
              Status: {getConnectionStatusText()}
            </Text>
          </View>
          
          {/* Permission Status */}
          <View style={styles.connectionRow}>
            <Ionicons 
              name={hasPermission ? "checkmark-circle" : "close-circle"} 
              size={16} 
              color={hasPermission ? "#4CAF50" : "#F44336"} 
            />
            <Text style={styles.statusText}>
              Microphone: {hasPermission === null ? 'Checking...' : hasPermission ? 'Granted' : 'Denied'}
            </Text>
          </View>
        </View>

        {/* Test Connection Button */}
        <TouchableOpacity 
          style={styles.testButton}
          onPress={testConnection}
        >
          <Ionicons name="wifi" size={16} color="white" />
          <Text style={styles.testButtonText}>Test Connection</Text>
        </TouchableOpacity>


        <TouchableOpacity
  style={[styles.exportButton, { backgroundColor: "#673AB7" }]}
  onPress={exportSessionsAsPDF}
>
  <Ionicons name="document-text-outline" size={18} color="white" />
  <Text style={styles.exportButtonText}>
    Export Summary as PDF
  </Text>
</TouchableOpacity>

      </View>

      {/* Recording Section */}
      <View style={styles.recordingSection}>
        <Text style={styles.sectionTitle}>Record Audio</Text>
        
        <View style={styles.recordingControls}>
          {isRecording ? (
            <>
              <View style={styles.recordingIndicator}>
                <View style={styles.recordingDot} />
                <Text style={styles.recordingText}>Recording...</Text>
                <Text style={styles.recordingTimer}>{formatDuration(recordingDuration)}</Text>
              </View>
              
              <TouchableOpacity 
                style={[styles.recordButton, styles.stopButton]} 
                onPress={stopRecording}
              >
                <Ionicons name="stop" size={24} color="white" />
                <Text style={styles.recordButtonText}>Stop Recording</Text>
              </TouchableOpacity>
            </>
          ) : (
            <TouchableOpacity 
              style={[
                styles.recordButton, 
                styles.startButton,
                hasPermission === false && styles.disabledButton
              ]} 
              onPress={startRecording}
              disabled={hasPermission === false}
            >
              <Ionicons name="mic" size={24} color="white" />
              <Text style={styles.recordButtonText}>
                {hasPermission === false ? 'Microphone Permission Denied' : 'Start Recording'}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {recordedUri && (
          <View style={styles.recordingInfo}>
            <Ionicons name="checkmark-circle" size={20} color="#4CAF50" />
            <Text style={styles.recordingInfoText}>
              Recording saved ({formatDuration(recordingDuration)})
            </Text>
          </View>
        )}
      </View>

      {/* File Selection */}
      <View style={styles.uploadSection}>
        <Text style={styles.sectionTitle}>Or Select Audio File</Text>
        
        <TouchableOpacity style={styles.uploadButton} onPress={pickAudioFile}>
          <Ionicons name="folder-open" size={24} color="white" />
          <Text style={styles.buttonText}>
            {selectedFile && !recordedUri ? 'Change Audio File' : 'Select Audio File'}
          </Text>
        </TouchableOpacity>

        {selectedFile?.assets?.[0] && (
          <View style={styles.fileInfo}>
            <Ionicons 
              name={recordedUri ? "mic" : "musical-notes"} 
              size={20} 
              color="#007AFF" 
            />
            <View style={styles.fileDetails}>
              <Text style={styles.fileText} numberOfLines={1}>
                {selectedFile.assets[0].name}
              </Text>
              {recordedUri && (
                <Text style={styles.fileDuration}>
                  Duration: {formatDuration(recordingDuration)}
                </Text>
              )}
            </View>
          </View>
        )}
      </View>

      {/* Note about file format */}
      {recordedUri && (
        <View style={styles.noteContainer}>
          <Ionicons name="information-circle" size={20} color="#FF9800" />
          <Text style={styles.noteText}>
            Note: Recordings are saved as M4A format. Your backend should handle this format.
          </Text>
        </View>
      )}

      {/* Analyze Button */}
      <TouchableOpacity 
        style={[
          styles.analyzeButton, 
          (!selectedFile || loading) && styles.disabledButton
        ]} 
        onPress={uploadAudio}
        disabled={!selectedFile || loading}
      >
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator color="#fff" />
            <Text style={styles.loadingText}>Analyzing...</Text>
          </View>
        ) : (
          <>
            <Ionicons name="analytics" size={20} color="white" />
            <Text style={styles.buttonText}>Analyze for Snoring</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Results Section */}
      {results && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>Analysis Results</Text>
          
          {/* Overall Summary */}
          <View style={styles.summaryCard}>
            <View style={styles.summaryHeader}>
              <Ionicons 
                name={getSnoringLevel(results.analysis.snoring_ratio).icon as any} 
                size={24} 
                color={getSnoringLevel(results.analysis.snoring_ratio).color} 
              />
              <Text style={styles.summaryTitle}>Overall Summary</Text>
            </View>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Snoring Level:</Text>
              <Text style={[styles.summaryValue, { color: getSnoringLevel(results.analysis.snoring_ratio).color }]}>
                {getSnoringLevel(results.analysis.snoring_ratio).text}
              </Text>
            </View>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Snoring Ratio:</Text>
              <Text style={styles.summaryValue}>
                {(results.analysis.snoring_ratio * 100).toFixed(1)}%
              </Text>
            </View>

            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Audio Duration:</Text>
              <Text style={styles.summaryValue}>
                {formatTime(results.audio_duration)}
              </Text>
            </View>
          </View>

          <View style={styles.summaryRow}>
  <Text style={styles.summaryLabel}>Quiet Minutes:</Text>
  <Text style={styles.summaryValue}>
    {formatTime(
      results.audio_duration -
      results.analysis.snoring_intervals.reduce(
        (sum: number, i: any) => sum + i.duration,
        0
      )
    )}
  </Text>
</View>


          {/* 🌙 Night Nudges (ADDED – display only) */}
{results.nudges && (
  <View style={styles.summaryCard}>
    <Text style={styles.summaryTitle}>Night Nudges</Text>

    <View style={styles.summaryRow}>
      <Text style={styles.summaryLabel}>Nudges Sent:</Text>
      <Text style={styles.summaryValue}>
        {results.nudges.nudges_this_hour}
      </Text>
    </View>

    {results.nudges.last_nudge_time && (
      <Text style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
        Last nudge at{" "}
        {new Date(results.nudges.last_nudge_time).toLocaleTimeString()}
      </Text>
    )}
  </View>
)}
{/* ✅ ADD EXPORT BUTTON RIGHT HERE */}
<TouchableOpacity
  style={styles.exportButton}
  onPress={exportSessionsAsCSV}
>
  <Ionicons name="download-outline" size={18} color="white" />
  <Text style={styles.exportButtonText}>
    Export Sessions as CSV
  </Text>
</TouchableOpacity>

          {/* Statistics */}
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{results.analysis.total_segments}</Text>
              <Text style={styles.statLabel}>Total Segments</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{results.analysis.snoring_segments}</Text>
              <Text style={styles.statLabel}>Snoring Segments</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{results.analysis.interval_count}</Text>
              <Text style={styles.statLabel}>Snoring Intervals</Text>
            </View>
          </View>

          {/* Snoring Intervals */}
          {results.analysis.snoring_intervals.length > 0 && (
            <View style={styles.intervalsContainer}>
              <Text style={styles.sectionTitle}>Snoring Intervals</Text>
              {results.analysis.snoring_intervals.map((interval: any, index: number) => (
                <View key={index} style={styles.intervalItem}>
                  <View style={styles.intervalHeader}>
                    <Ionicons name="time" size={16} color="#666" />
                    <Text style={styles.intervalTitle}>Interval {index + 1}</Text>
                  </View>
                  <Text style={styles.intervalText}>
                    {interval.start_time}s - {interval.end_time}s ({interval.duration}s)
                  </Text>
                </View>
              ))}
            </View>
          )}

          {/* Segment Predictions */}
          {results.segment_predictions && results.segment_predictions.length > 0 && (
            <View style={styles.segmentsContainer}>
              <Text style={styles.sectionTitle}>Segment Analysis</Text>
              {results.segment_predictions.slice(0, 8).map((pred: any, index: number) => (
                <View key={index} style={[
                  styles.segmentItem,
                  pred.class === 'Snoring' ? styles.snoringSegment : styles.normalSegment
                ]}>
                  <Text style={styles.segmentText}>
                    Segment {pred.segment}: {pred.class}
                  </Text>
                  <Text style={styles.confidenceText}>
                    {(pred.confidence * 100).toFixed(1)}% confidence
                  </Text>
                </View>
              ))}
            </View>
          )}

          {/* Conclusion Message */}
          <View style={styles.conclusionCard}>
            <Text style={styles.conclusionText}>{results.message}</Text>
          </View>
        </View>
      )}

      {/* Visualizations */}
      {visualizations && (
        <View style={styles.visualizationsContainer}>
          <Text style={styles.visualizationsTitle}>Analysis Visualizations</Text>
          
          {visualizations.analysis_plot && (
            <View style={styles.visualizationCard}>
              <Text style={styles.visualizationTitle}>Comprehensive Analysis</Text>
              <Image 
                source={base64ToImageSource(visualizations.analysis_plot)}
                style={styles.visualizationImage}
                resizeMode="contain"
              />
              <Text style={styles.visualizationDescription}>
                Shows audio waveform, snoring probability, detection timeline, and distribution
              </Text>
            </View>
          )}
          
          {visualizations.statistics_plot && (
            <View style={styles.visualizationCard}>
              <Text style={styles.visualizationTitle}>Statistics & Metrics</Text>
              <Image 
                source={base64ToImageSource(visualizations.statistics_plot)}
                style={styles.visualizationImage}
                resizeMode="contain"
              />
              <Text style={styles.visualizationDescription}>
                Detailed statistics, confidence distribution, and interval analysis
              </Text>
            </View>
          )}
          
          {visualizations.timeline_plot && (
            <View style={styles.visualizationCard}>
              <Text style={styles.visualizationTitle}>Snoring Timeline</Text>
              <Image 
                source={base64ToImageSource(visualizations.timeline_plot)}
                style={styles.visualizationImage}
                resizeMode="contain"
              />
              <Text style={styles.visualizationDescription}>
                Clear timeline showing snoring patterns throughout the recording
              </Text>
            </View>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'white',
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 10,
    color: '#333',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 5,
    marginBottom: 15,
  },
  connectionSection: {
    width: '100%',
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  connectionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  connectionText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 5,
  },
  statusIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 5,
  },
  testButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    gap: 6,
  },
  testButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  // Recording Section Styles
  recordingSection: {
    backgroundColor: 'white',
    padding: 20,
    marginHorizontal: 10,
    borderRadius: 12,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  recordingControls: {
    alignItems: 'center',
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
    backgroundColor: '#fff5f5',
    padding: 12,
    borderRadius: 8,
    width: '100%',
    justifyContent: 'center',
    gap: 10,
  },
  recordingDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#ff4444',
  },
  recordingText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#ff4444',
  },
  recordingTimer: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  recordButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    width: '100%',
    gap: 10,
  },
  startButton: {
    backgroundColor: '#ff4444',
  },
  stopButton: {
    backgroundColor: '#666',
  },
  recordButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  recordingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f0fff0',
    borderRadius: 8,
    gap: 8,
  },
  recordingInfoText: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: '500',
  },
  // File Upload Section
  uploadSection: {
    backgroundColor: 'white',
    padding: 20,
    marginHorizontal: 10,
    borderRadius: 12,
    marginBottom: 10,
  },
  uploadButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  fileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    gap: 10,
  },
  fileDetails: {
    flex: 1,
  },
  fileText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  fileDuration: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  // Note container
  noteContainer: {
    backgroundColor: '#FFF3E0',
    padding: 12,
    marginHorizontal: 10,
    marginBottom: 10,
    borderRadius: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  noteText: {
    fontSize: 12,
    color: '#E65100',
    flex: 1,
  },
  // Analyze Button
  analyzeButton: {
    backgroundColor: '#34C759',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    margin: 10,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  loadingText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  // Results Section
  resultsContainer: {
    padding: 10,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 15,
    color: '#333',
  },
  summaryCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 10,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 5,
  },
  summaryLabel: {
    fontSize: 16,
    color: '#666',
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
    gap: 10,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
    textAlign: 'center',
  },
  intervalsContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  intervalItem: {
    padding: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#ff4444',
    marginBottom: 8,
    backgroundColor: '#fff5f5',
    borderRadius: 8,
  },
  intervalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
    gap: 5,
  },
  intervalTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  intervalText: {
    fontSize: 12,
    color: '#666',
  },
  segmentsContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  segmentItem: {
    padding: 10,
    borderRadius: 6,
    marginBottom: 5,
  },
  snoringSegment: {
    backgroundColor: '#fff5f5',
    borderLeftWidth: 4,
    borderLeftColor: '#ff4444',
  },
  normalSegment: {
    backgroundColor: '#f8f9fa',
    borderLeftWidth: 4,
    borderLeftColor: '#34C759',
  },
  segmentText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceText: {
    fontSize: 10,
    color: '#666',
    marginTop: 2,
  },
  conclusionCard: {
    backgroundColor: '#e3f2fd',
    padding: 15,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#2196f3',
  },
  conclusionText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1976d2',
    textAlign: 'center',
  },
  // Visualization Styles
  visualizationsContainer: {
    marginTop: 20,
    padding: 10,
  },
  visualizationsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 15,
    color: '#333',
  },
  visualizationCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  visualizationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
    textAlign: 'center',
  },
  visualizationImage: {
    width: '100%',
    height: 300,
    borderRadius: 8,
  },
  visualizationDescription: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    fontStyle: 'italic',
  },

  exportButton: {
  backgroundColor: '#007AFF',
  padding: 14,
  borderRadius: 10,
  marginVertical: 10,
  flexDirection: 'row',
  justifyContent: 'center',
  alignItems: 'center',
  gap: 8,
},

exportButtonText: {
  color: 'white',
  fontSize: 15,
  fontWeight: 'bold',
},

});