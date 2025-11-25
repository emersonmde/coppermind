//! Formatting utilities for human-readable output.
//!
//! This module provides consistent formatting for timestamps, durations,
//! and other display values across the UI.

/// Format Unix timestamp to human-readable relative time (e.g., "5 mins ago").
///
/// # Arguments
///
/// * `timestamp` - Unix timestamp in seconds since epoch
///
/// # Returns
///
/// A human-readable string like "Just now", "5 mins ago", "2 hours ago", etc.
///
/// # Examples
///
/// ```ignore
/// use coppermind::utils::formatting::format_timestamp;
///
/// let now = std::time::SystemTime::now()
///     .duration_since(std::time::UNIX_EPOCH)
///     .unwrap()
///     .as_secs();
///
/// assert_eq!(format_timestamp(now), "Just now");
/// assert_eq!(format_timestamp(now - 120), "2 mins ago");
/// ```
pub fn format_timestamp(timestamp: u64) -> String {
    if timestamp == 0 {
        return "Unknown".to_string();
    }

    #[cfg(target_arch = "wasm32")]
    {
        let now = instant::SystemTime::now()
            .duration_since(instant::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let elapsed = now.saturating_sub(timestamp);
        format_duration(elapsed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let elapsed = now.saturating_sub(timestamp);
        format_duration(elapsed)
    }
}

/// Format duration in seconds to human-readable string.
///
/// # Arguments
///
/// * `seconds` - Duration in seconds
///
/// # Returns
///
/// A human-readable string like "Just now", "5 mins ago", "2 hours ago", etc.
///
/// # Examples
///
/// ```ignore
/// use coppermind::utils::formatting::format_duration;
///
/// assert_eq!(format_duration(30), "Just now");
/// assert_eq!(format_duration(120), "2 mins ago");
/// assert_eq!(format_duration(7200), "2 hours ago");
/// assert_eq!(format_duration(172800), "2 days ago");
/// ```
pub fn format_duration(seconds: u64) -> String {
    match seconds {
        0..=59 => "Just now".to_string(),
        60..=3599 => {
            let mins = seconds / 60;
            if mins == 1 {
                "1 min ago".to_string()
            } else {
                format!("{} mins ago", mins)
            }
        }
        3600..=86399 => {
            let hours = seconds / 3600;
            if hours == 1 {
                "1 hour ago".to_string()
            } else {
                format!("{} hours ago", hours)
            }
        }
        86400..=2591999 => {
            let days = seconds / 86400;
            if days == 1 {
                "1 day ago".to_string()
            } else {
                format!("{} days ago", days)
            }
        }
        _ => {
            let days = seconds / 86400;
            format!("{} days ago", days)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_just_now() {
        assert_eq!(format_duration(0), "Just now");
        assert_eq!(format_duration(30), "Just now");
        assert_eq!(format_duration(59), "Just now");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60), "1 min ago");
        assert_eq!(format_duration(120), "2 mins ago");
        assert_eq!(format_duration(3599), "59 mins ago");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3600), "1 hour ago");
        assert_eq!(format_duration(7200), "2 hours ago");
        assert_eq!(format_duration(86399), "23 hours ago");
    }

    #[test]
    fn test_format_duration_days() {
        assert_eq!(format_duration(86400), "1 day ago");
        assert_eq!(format_duration(172800), "2 days ago");
        assert_eq!(format_duration(2591999), "29 days ago");
    }

    #[test]
    fn test_format_duration_many_days() {
        assert_eq!(format_duration(2592000), "30 days ago");
        assert_eq!(format_duration(31536000), "365 days ago");
    }

    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(format_timestamp(0), "Unknown");
    }
}
