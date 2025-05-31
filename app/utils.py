import datetime
import logging
import csv
import io

logger = logging.getLogger("parakeet-asr.utils") # Use a consistent logger name

def format_srt_time(seconds: float) -> str:
    """Converts seconds to SRT time format (HH:MM:SS,ms)"""
    try:
        # Ensure seconds is not negative, which can happen with slight miscalculations
        sanitized_total_seconds = max(0.0, seconds)
        delta = datetime.timedelta(seconds=sanitized_total_seconds)
        
        # Extract total integer seconds to avoid issues with direct delta.seconds for large values
        total_int_seconds = int(delta.total_seconds())

        hours = total_int_seconds // 3600
        remainder_seconds_after_hours = total_int_seconds % 3600
        minutes = remainder_seconds_after_hours // 60
        seconds_part = remainder_seconds_after_hours % 60
        
        # Microseconds are part of the remainder not covered by total_seconds for timedelta
        milliseconds = delta.microseconds // 1000
        
        return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"
    except Exception as e:
        logger.error(f"Error formatting SRT time for {seconds}s: {e}", exc_info=True)
        # Fallback to a default error timestamp
        return "00:00:00,000"


def generate_srt_content(segments: list[dict]) -> str:
    """
    Generates SRT content from a list of segment dictionaries.
    Each dictionary should have 'start', 'end', and 'text'.
    """
    srt_content_parts = []
    if not isinstance(segments, list):
        logger.error(f"generate_srt_content expects a list of segments, got {type(segments)}")
        return ""

    for i, segment in enumerate(segments):
        try:
            if not isinstance(segment, dict):
                logger.warning(f"Skipping non-dictionary segment in SRT generation: {segment}")
                continue

            start_time_str = format_srt_time(float(segment.get("start", 0.0)))
            end_time_str = format_srt_time(float(segment.get("end", 0.0)))
            # Sanitize text: remove newlines within a segment for SRT compatibility
            segment_text = str(segment.get("text", "")).replace('\n', ' ').strip()
            
            srt_content_parts.append(str(i + 1))
            srt_content_parts.append(f"{start_time_str} --> {end_time_str}")
            srt_content_parts.append(segment_text)
            srt_content_parts.append("") # Blank line separator
        except KeyError as ke: # Should be caught by .get() with default, but good practice
            logger.error(f"Segment missing expected key {ke} for SRT generation: {segment}")
            continue 
        except (ValueError, TypeError) as ve:
            logger.error(f"Invalid type for start/end time in segment for SRT: {segment}. Error: {ve}")
            continue
        except Exception as e:
            logger.error(f"Error processing segment for SRT: {segment}. Error: {e}", exc_info=True)
            continue

    return "\n".join(srt_content_parts)


def generate_csv_content(segments: list[dict]) -> str:
    """
    Generates CSV formatted string content from a list of segment dictionaries.
    Each dictionary should ideally have 'start', 'end', and 'text'.
    """
    if not isinstance(segments, list):
        logger.error(f"generate_csv_content expects a list of segments, got {type(segments)}")
        return "Error: Invalid segment data format."

    try:
        csv_output = io.StringIO()
        # Define CSV fieldnames for clarity and consistency
        # Ensure these match the keys you expect in your segment dictionaries,
        # or use segment.get("key", default_value) when writing rows.
        fieldnames = ["Start (s)", "End (s)", "Text"]
        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)

        csv_writer.writeheader() # Write the header row

        for segment in segments:
            if not isinstance(segment, dict):
                logger.warning(f"Skipping non-dictionary segment in CSV generation: {segment}")
                continue
            try:
                # Prepare row data, using .get() for safety and converting to string where appropriate
                row_data = {
                    "Start (s)": f"{float(segment.get('start', 0.0)):.3f}",
                    "End (s)": f"{float(segment.get('end', 0.0)):.3f}",
                    "Text": str(segment.get("text", "")).strip()
                }
                csv_writer.writerow(row_data)
            except (ValueError, TypeError) as ve:
                logger.error(f"Invalid type for start/end/text in segment for CSV: {segment}. Error: {ve}")
                # Optionally write a placeholder error row or skip
                # csv_writer.writerow({"Start (s)": "ERROR", "End (s)": "ERROR", "Text": f"Malformed segment data: {ve}"})
                continue
            except Exception as e_row:
                 logger.error(f"Error writing segment row to CSV: {segment}. Error: {e_row}", exc_info=True)
                 continue


        csv_content_str = csv_output.getvalue()
        return csv_content_str
    except Exception as e_csv:
        logger.error(f"General error generating CSV content: {e_csv}", exc_info=True)
        return "Error generating CSV content." # Fallback error string
    finally:
        if 'csv_output' in locals() and csv_output:
            csv_output.close()
