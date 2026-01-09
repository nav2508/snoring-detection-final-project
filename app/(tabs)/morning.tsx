// @ts-nocheck
import { Ionicons } from "@expo/vector-icons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useEffect, useState } from "react";
import {
    Alert,
    ScrollView,
    StyleSheet,
    Text,
    View
} from "react-native";






const getLastNSessions = (sessions: any[], n = 7) => {
  if (!sessions || sessions.length === 0) return [];
  return sessions.slice(Math.max(sessions.length - n, 0));
};

const calculateWeeklyTrends = (sessions: any[]) => {
  if (!sessions || sessions.length === 0) return null;

  const total = sessions.length;

  const avgSnoringRatio =
    sessions.reduce((sum, s) => sum + (s.snoringRatio ?? 0), 0) / total;

  const avgQuietMinutes =
    sessions.reduce((sum, s) => sum + (s.quietMinutes ?? 0), 0) / total;

  const avgNudges =
    sessions.reduce(
      (sum, s) => sum + (s.nudges?.nudges_this_hour ?? 0),
      0
    ) / total;

  return {
    avgSnoringRatio,
    avgQuietMinutes,
    avgNudges,
    nights: total,
  };
};

const calculateMonthlyTrends = (sessions: any[]) => {
  if (!sessions || sessions.length === 0) return null;

  const total = sessions.length;

  const avgSnoringRatio =
    sessions.reduce((sum, s) => sum + (s.snoringRatio ?? 0), 0) / total;

  const avgQuietMinutes =
    sessions.reduce((sum, s) => sum + (s.quietMinutes ?? 0), 0) / total;

  const avgNudges =
    sessions.reduce(
      (sum, s) => sum + (s.nudges?.nudges_this_hour ?? 0),
      0
    ) / total;

  return {
    avgSnoringRatio,
    avgQuietMinutes,
    avgNudges,
    nights: total,
  };
};

const normalizeTimelineSegments = (
  intervals: any[],
  totalDuration: number
) => {
  if (!intervals || !totalDuration) return [];

  return intervals.map((interval) => {
    const startPct = (interval.start / totalDuration) * 100;
    const widthPct =
      ((interval.end - interval.start) / totalDuration) * 100;

    return {
      startPct,
      widthPct,
    };
  });
};



const calculateBadges = (sessions: any[], weeklyTrends: any) => {
  const badges: string[] = [];

  // 🏅 Three Quiet Nights in a Row
  if (sessions.length >= 3) {
    const last3 = sessions.slice(-3);
    const quietStreak = last3.every(
      (s) => s.snoringRatio !== undefined && s.snoringRatio < 0.2
    );

    if (quietStreak) {
      badges.push("🏅 Three Quiet Nights in a Row");
    }
  }

  // 🏅 Low-Nudge Week
  if (weeklyTrends && weeklyTrends.avgNudges < 1) {
    badges.push("🏅 Low-Nudge Week");
  }

  return badges;
};

const getPercent = (time: number, total: number) => {
  if (!total || total === 0) return 0;
  return Math.min(Math.max((time / total) * 100, 0), 100);
};
const TimelineBar = ({ session }: { session: any }) => {
  const duration = session.audioDuration ?? 0;

  const snoringIntervals = session.snoringIntervals ?? [];
  const nudges = session.nudges?.timestamps ?? [];

  return (
    <View style={styles.timelineContainer}>
      {/* Base bar */}
      <View style={styles.timelineBase} />

      {/* Snoring segments */}
      {snoringIntervals.map((interval: any, idx: number) => {
        const left = getPercent(interval.start, duration);
        const width = getPercent(interval.end - interval.start, duration);

        return (
          <View
            key={`snore-${idx}`}
            style={[
              styles.snoreBlock,
              { left: `${left}%`, width: `${width}%` },
            ]}
          />
        );
      })}

      {/* Nudge markers */}
      {nudges.map((t: number, idx: number) => {
        const left = getPercent(t, duration);
        return (
          <View
            key={`nudge-${idx}`}
            style={[styles.nudgeDot, { left: `${left}%` }]}
          />
        );
      })}
    </View>
  );
};

const BASE_REFLECTION_QUESTIONS = [
  { id: "caffeine", text: "Did you have caffeine after 6 PM?", type: "yes_no" },
  { id: "alcohol", text: "Did you consume alcohol last evening?", type: "yes_no" },
  { id: "screen", text: "Screen use within 1 hour of bedtime?", type: "yes_no" },
  { id: "stress", text: "How stressed did you feel before sleep?", type: "scale" },
];

const getAdaptiveQuestions = (latest: any) => {
  const qs = [...BASE_REFLECTION_QUESTIONS];

  if (latest?.snoringRatio > 0.4) {
    qs.unshift({
      id: "back_sleeping",
      text: "Did you mostly sleep on your back?",
      type: "yes_no",
    });
  }

  return qs;
};


const generateInsight = (latest: any) => {
  if (!latest?.reflection) return null;

  if (latest.reflection.caffeine && latest.snoringRatio > 0.4) {
    return "Late caffeine may have contributed to increased snoring.";
  }

  if (latest.reflection.back_sleeping && latest.nudges?.nudges_this_hour > 0) {
    return "Back sleeping is associated with more nudges.";
  }

  if (latest.quietMinutes > 10) {
    return "You had a relatively quiet night — keep this routine!";
  }

  return null;
};

const generateWeeklyCorrelation = (sessions: any[]) => {
  if (sessions.length < 3) return null;

  const caffeineNights = sessions.filter(s => s.reflection?.caffeine);
  if (caffeineNights.length < 2) return null;

  const avgSnore =
    caffeineNights.reduce((sum, s) => sum + (s.snoringRatio ?? 0), 0) /
    caffeineNights.length;

  return avgSnore > 0.4
    ? "Caffeine nights show higher snoring on average."
    : null;
};

const getTonightSuggestion = (latest: any) => {
  if (latest.snoringRatio > 0.5) {
    return "Try sleeping on your side tonight.";
  }

  if (latest.reflection?.caffeine) {
    return "Avoid caffeine after 6 PM tonight.";
  }

  if (latest.quietMinutes < 5) {
    return "Aim for a calmer bedtime routine.";
  }

  return "Keep following your current routine.";
};




export default function MorningReviewScreen() {
  const [latest, setLatest] = useState<any>(null);
  const [loading, setLoading] = useState(true);
    const [weeklyTrends, setWeeklyTrends] = useState<any>(null);

    const [allSessions, setAllSessions] = useState<any[]>([]);

    const [monthlyTrends, setMonthlyTrends] = useState<any>(null);

      const [badges, setBadges] = useState<string[]>([]);

      const [reflectionAnswers, setReflectionAnswers] = useState<any>({});

      const [lockedQuestions, setLockedQuestions] = useState<Record<string, boolean>>({});





  useEffect(() => {
    const loadLatest = async () => {
      try {
        const raw = await AsyncStorage.getItem("night_sessions");
        if (!raw) {
          setLatest(null);
          setLoading(false);
          return;
        }

        const sessions = JSON.parse(raw);

        setAllSessions(sessions);

        if (!Array.isArray(sessions) || sessions.length === 0) {
          setLatest(null);
          setLoading(false);
          return;
        }

        const lastSession = sessions[sessions.length - 1];
        setLatest(lastSession);

        setReflectionAnswers(lastSession.reflection || {});

        // ✅ Do NOT lock on load
setLockedQuestions({});



              // ✅ Weekly trends (last 7 sessions)
      const last7 = getLastNSessions(sessions, 7);
      const trends = calculateWeeklyTrends(last7);
      setWeeklyTrends(trends);

      // ✅ Monthly trends (last 30 sessions)
const last30 = getLastNSessions(sessions, 30);
const monthly = calculateMonthlyTrends(last30);
setMonthlyTrends(monthly);


      const earnedBadges = calculateBadges(sessions, trends);
setBadges(earnedBadges);


      } catch (e) {
        console.log("❌ Failed to load sessions", e);
        Alert.alert("Error", "Could not load your saved sessions.");
      } finally {
        setLoading(false);
      }
    };

    loadLatest();
  }, []);

   const saveReflection = async (key: string, value: any) => {
  const raw = await AsyncStorage.getItem("night_sessions");
  if (!raw) return;

  // 🔒 lock now (safe, because we know we can persist)
  setLockedQuestions(prev => ({ ...prev, [key]: true }));

  const updated = { ...reflectionAnswers, [key]: value };

  setReflectionAnswers(updated);

  setLatest((prev: any) => ({
    ...prev,
    reflection: updated,
  }));

  const sessions = JSON.parse(raw);
  sessions[sessions.length - 1].reflection = updated;

  await AsyncStorage.setItem("night_sessions", JSON.stringify(sessions));
};


  


  const formatTime = (seconds: number) => {
    if (!seconds && seconds !== 0) return "—";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const formatMinutes = (minutes: number) => {
    if (minutes === null || minutes === undefined) return "—";
    return `${minutes} min`;
  };

  if (loading) {
    return (
      <View style={styles.center}>
        <Ionicons name="hourglass-outline" size={26} color="#007AFF" />
        <Text style={styles.centerText}>Loading your last night summary...</Text>
      </View>
    );
  }

  if (!latest) {
    return (
      <View style={styles.center}>
        <Ionicons name="cloud-offline-outline" size={30} color="#999" />
        <Text style={styles.centerTitle}>No sessions yet</Text>
        <Text style={styles.centerText}>
          Record or upload once in Snoring Analysis.
          {"\n"}Your last night summary will appear here.
        </Text>
      </View>
    );
  }

  const snoringPct =
    latest.snoringRatio !== undefined && latest.snoringRatio !== null
      ? `${(latest.snoringRatio * 100).toFixed(1)}%`
      : "—";

  const nudgesSent =
    latest.nudges?.nudges_this_hour !== undefined && latest.nudges?.nudges_this_hour !== null
      ? latest.nudges.nudges_this_hour
      : 0;
const timelineSegments = normalizeTimelineSegments(
  latest.snoringIntervals || [],
  latest.audioDuration
);


  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Ionicons name="sunny-outline" size={30} color="#007AFF" />
        <Text style={styles.title}>Morning Review</Text>
        <Text style={styles.subtitle}>
          Your most recent session summary (even if you don’t record today)
        </Text>
      </View>

      {/* Summary Card */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Last Night Summary</Text>

        <View style={styles.row}>
          <Text style={styles.label}>Date</Text>
          <Text style={styles.value}>
            {latest.startTime ? new Date(latest.startTime).toLocaleString() : "—"}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Sleep Duration</Text>
          <Text style={styles.value}>{formatTime(latest.audioDuration)}</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Snoring %</Text>
          <Text style={styles.value}>{snoringPct}</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Quiet Snore Minutes</Text>
          <Text style={styles.value}>{formatMinutes(latest.quietMinutes)}</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Nudges Sent</Text>
          <Text style={styles.value}>{nudgesSent}</Text>
        </View>
      </View>

      {/* 🕒 Overnight Timeline */}
<View style={styles.card}>
  <Text style={styles.cardTitle}>Overnight Timeline</Text>

  {/* Timeline bar */}
  <View style={styles.timelineContainer}>
    {/* Quiet background */}
    <View style={styles.timelineBackground} />

    {/* Snoring segments */}
    {timelineSegments.map((seg, idx) => (
      <View
        key={idx}
        style={[
          styles.snoringSegment,
          {
            left: `${seg.startPct}%`,
            width: `${seg.widthPct}%`,
          },
        ]}
      />
    ))}

    {/* Nudge markers */}
    {(latest.nudges?.timestamps || []).map((t: number, idx: number) => {
      const leftPct = (t / latest.audioDuration) * 100;
      return (
        <View
          key={`nudge-${idx}`}
          style={[
            styles.nudgeMarker,
            { left: `${leftPct}%` },
          ]}
        />
      );
    })}
  </View>

  {/* Legend */}
  <View style={styles.legendRow}>
    <View style={[styles.legendDot, { backgroundColor: "#e53935" }]} />
    <Text style={styles.legendText}>Snoring</Text>

    <View style={[styles.legendDot, { backgroundColor: "#ccc" }]} />
    <Text style={styles.legendText}>Quiet</Text>

    <View style={[styles.legendDot, { backgroundColor: "#1976d2" }]} />
    <Text style={styles.legendText}>Nudges</Text>
  </View>
</View>





            {/* Weekly Trends Card */}
      {weeklyTrends && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            Weekly Trends (Last {weeklyTrends.nights} nights)
          </Text>

          <View style={styles.row}>
            <Text style={styles.label}>Avg Snoring %</Text>
            <Text style={styles.value}>
              {(weeklyTrends.avgSnoringRatio * 100).toFixed(1)}%
            </Text>
          </View>

          <View style={styles.row}>
            <Text style={styles.label}>Avg Quiet Minutes</Text>
            <Text style={styles.value}>
              {weeklyTrends.avgQuietMinutes.toFixed(1)} min
            </Text>
          </View>

          <View style={styles.row}>
            <Text style={styles.label}>Avg Nudges / Night</Text>
            <Text style={styles.value}>
              {weeklyTrends.avgNudges.toFixed(1)}
            </Text>
          </View>
        </View>
      )}

      {generateWeeklyCorrelation(getLastNSessions(allSessions, 7)) && (
  <View style={styles.card}>
    <Text style={styles.cardTitle}>Habit Pattern</Text>
    <Text style={{ fontSize: 14 }}>
      {generateWeeklyCorrelation(getLastNSessions(allSessions, 7))}
    </Text>
  </View>
)}


      {/* Monthly Trends Card */}
{monthlyTrends && (
  <View style={styles.card}>
    <Text style={styles.cardTitle}>
      Monthly Trends (Last {monthlyTrends.nights} nights)
    </Text>

    <View style={styles.row}>
      <Text style={styles.label}>Avg Snoring %</Text>
      <Text style={styles.value}>
        {(monthlyTrends.avgSnoringRatio * 100).toFixed(1)}%
      </Text>
    </View>

    <View style={styles.row}>
      <Text style={styles.label}>Avg Quiet Minutes</Text>
      <Text style={styles.value}>
        {monthlyTrends.avgQuietMinutes.toFixed(1)} min
      </Text>
    </View>

    <View style={styles.row}>
      <Text style={styles.label}>Avg Nudges / Night</Text>
      <Text style={styles.value}>
        {monthlyTrends.avgNudges.toFixed(1)}
      </Text>
    </View>
  </View>
)}


      {/* 🏅 Badges Card */}
{badges.length > 0 && (
  <View style={styles.card}>
    <Text style={styles.cardTitle}>Achievements</Text>

    {badges.map((badge, index) => (
      <View key={index} style={{ paddingVertical: 6 }}>
        <Text style={{ fontSize: 15, fontWeight: "600", color: "#2e7d32" }}>
          {badge}
        </Text>
      </View>
    ))}
  </View>
)}

<View style={styles.card}>
  <Text style={styles.cardTitle}>Quick Reflection</Text>

  {getAdaptiveQuestions(latest).map(q => (
    <View key={q.id} style={{ marginBottom: 12 }}>
      <Text style={{ fontSize: 14 }}>{q.text}</Text>

      {q.type === "yes_no" && (
  <View
    style={{ flexDirection: "row", marginTop: 6 }}
    
  >

          <Text
  onPress={() => {
    if (lockedQuestions[q.id]) return;
    saveReflection(q.id, true);
  }}
  suppressHighlighting
  style={[
    styles.choice,
    styles.choiceTouchable,
    reflectionAnswers[q.id] === true && styles.choiceSelected
  ]}
>
  Yes
</Text>



<Text
  onPress={() => {
    if (lockedQuestions[q.id]) return;
    saveReflection(q.id, false);
  }}
  suppressHighlighting
  style={[
    styles.choice,
    styles.choiceTouchable,
    reflectionAnswers[q.id] === false && styles.choiceSelected
  ]}
>
  No
</Text>



        </View>
      )}

      {q.type === "scale" && (
  <View
    style={{ flexDirection: "row", marginTop: 6 }}
    
  >

          {[1,2,3,4,5].map(n => (
            <Text
  key={n}
  onPress={() => {
    if (lockedQuestions[q.id]) return;
    saveReflection(q.id, n);
  }}
  suppressHighlighting
  style={[
    styles.scale,
    styles.scaleTouchable,
    reflectionAnswers[q.id] === n && styles.scaleSelected
  ]}
>
  {n}
</Text>




          ))}
        </View>
      )}
    </View>
  ))}
</View>

{generateInsight(latest) && (
  <View style={styles.card}>
    <Text style={styles.cardTitle}>Insight</Text>
    <Text style={{ fontSize: 14, color: "#444" }}>
      {generateInsight(latest)}
    </Text>
  </View>
)}

<View style={styles.card}>
  <Text style={styles.cardTitle}>Try This Tonight</Text>
  <Text style={{ fontSize: 14 }}>{getTonightSuggestion(latest)}</Text>
</View>




      {/* Optional: Preferences card */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Preferences Used</Text>

        <View style={styles.row}>
          <Text style={styles.label}>Quiet Hours</Text>
          <Text style={styles.value}>
            {latest.preferences?.quiet_start || "—"} → {latest.preferences?.quiet_end || "—"}
          </Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Sensitivity</Text>
          <Text style={styles.value}>{latest.preferences?.sensitivity || "—"}</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Max Nudges / Hour</Text>
          <Text style={styles.value}>{latest.preferences?.max_nudges_per_hour ?? "—"}</Text>
        </View>

        <View style={styles.row}>
          <Text style={styles.label}>Cooldown</Text>
          <Text style={styles.value}>{latest.preferences?.cooldown_minutes ?? "—"} min</Text>
        </View>
      </View>

      {/* Footer hint */}
      <Text style={styles.footer}>
        Tip: This screen reads from saved sessions in local storage (night_sessions).
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f5f5f5" },

  header: {
    backgroundColor: "white",
    padding: 18,
    alignItems: "center",
    marginBottom: 10,
  },
  title: { fontSize: 22, fontWeight: "bold", color: "#333", marginTop: 8 },
  subtitle: { fontSize: 13, color: "#666", marginTop: 6, textAlign: "center" },

  card: {
    backgroundColor: "white",
    marginHorizontal: 10,
    marginBottom: 10,
    padding: 16,
    borderRadius: 12,
    shadowColor: "#000",
    shadowOpacity: 0.08,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: { fontSize: 16, fontWeight: "bold", color: "#333", marginBottom: 10 },

  row: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 6,
    borderBottomWidth: 0.5,
    borderBottomColor: "#eee",
  },
  label: { fontSize: 14, color: "#666" },
  value: { fontSize: 14, fontWeight: "bold", color: "#333", maxWidth: "62%", textAlign: "right" },

  footer: {
    textAlign: "center",
    fontSize: 12,
    color: "#777",
    marginTop: 6,
    marginBottom: 25,
  },

  center: {
    flex: 1,
    backgroundColor: "#f5f5f5",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  centerTitle: { fontSize: 18, fontWeight: "bold", color: "#333", marginTop: 10 },
  centerText: { fontSize: 13, color: "#666", textAlign: "center", marginTop: 8 },

  timelineContainer: {
  height: 26,
  marginTop: 10,
  position: "relative",
  borderRadius: 6,
  overflow: "hidden",
},

timelineBase: {
  position: "absolute",
  height: "100%",
  width: "100%",
  backgroundColor: "#e0e0e0",
  borderRadius: 6,
},

snoreBlock: {
  position: "absolute",
  height: "100%",
  backgroundColor: "#e53935",
  borderRadius: 6,
},

nudgeDot: {
  position: "absolute",
  top: -4,
  width: 10,
  height: 10,
  borderRadius: 5,
  backgroundColor: "#1e88e5",
},

timelineContainer: {
  height: 24,
  marginTop: 10,
  marginBottom: 10,
  position: "relative",
  borderRadius: 12,
  overflow: "hidden",
},

timelineBackground: {
  position: "absolute",
  height: "100%",
  width: "100%",
  backgroundColor: "#e0e0e0",
  borderRadius: 12,
},

snoringSegment: {
  position: "absolute",
  height: "100%",
  backgroundColor: "#e53935",
},

nudgeMarker: {
  position: "absolute",
  height: "100%",
  width: 3,
  backgroundColor: "#1976d2",
},

legendRow: {
  flexDirection: "row",
  alignItems: "center",
  marginTop: 8,
},

legendDot: {
  width: 10,
  height: 10,
  borderRadius: 5,
  marginRight: 4,
  marginLeft: 12,
},

legendText: {
  fontSize: 12,
  color: "#555",
},

choice: {
  marginRight: 16,
  marginTop: 4,
  color: "#1976d2",
  fontWeight: "600",
},

scale: {
  marginRight: 10,
  padding: 6,
  borderRadius: 4,
  backgroundColor: "#eee",
},

choiceSelected: {
  color: "#fff",
  backgroundColor: "#1976d2",
  paddingHorizontal: 10,
  paddingVertical: 4,
  borderRadius: 6,
},

scaleSelected: {
  backgroundColor: "#1976d2",
  color: "#fff",
  fontWeight: "700",
},

choiceTouchable: {
  paddingHorizontal: 16,
  paddingVertical: 10,
  borderRadius: 8,
  backgroundColor: "#e3f2fd",
  overflow: "hidden",
},

scaleTouchable: {
  paddingHorizontal: 14,
  paddingVertical: 10,
  borderRadius: 6,
},






});
