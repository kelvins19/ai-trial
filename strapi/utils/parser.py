import re
import datetime
import time
from dateutil import parser

def parse_event_dates(date_string):
    """
    Parse event date strings and return start_date and end_date as Unix timestamps
    
    Args:
        date_string (str): Event date string in various formats
        
    Returns:
        dict: Dictionary with start_date and end_date as Unix timestamps
    """
    # Get today's date and end of year date
    today = datetime.datetime.now()
    end_of_year = datetime.datetime(today.year, 12, 31)
    
    # Default return values (today to end of year)
    result = {
        "start_date": int(today.timestamp()),
        "end_date": int(end_of_year.timestamp())
    }
    
    # Handle case: No date information
    if not date_string or date_string.strip() == "":
        return result
    
    # Handle case: Ongoing events
    if date_string.lower() in ["now on till further notice", "ongoing"]:
        return result
    
    # Case: Specific days of the week in a month (e.g., "6, 13, 20 & 27 Mar 2025 (every Thursday)")
    specific_days_pattern = r'(\d+(?:,\s*\d+)*(?:\s*&\s*\d+)?)\s+([A-Za-z]+)\s+(\d{4})'
    specific_days_match = re.search(specific_days_pattern, date_string)
    
    if specific_days_match:
        days = re.findall(r'\d+', specific_days_match.group(1))
        month = specific_days_match.group(2)
        year = specific_days_match.group(3)
        
        all_dates = []
        for day in days:
            try:
                date_obj = parser.parse(f"{day} {month} {year}")
                all_dates.append(date_obj)
            except:
                continue
        
        if all_dates:
            all_dates.sort()
            result["start_date"] = int(all_dates[0].timestamp())
            result["end_date"] = int(all_dates[-1].timestamp())
            return result
    
    # Case: Date range with days of week (e.g., "17 Feb - 27 Mar 2025 (Monday-Thursday)")
    range_pattern = r'(\d+\s+[A-Za-z]+)(?:\s+(\d{4}))?\s*-\s*(\d+\s+[A-Za-z]+)\s+(\d{4})'
    range_match = re.search(range_pattern, date_string)
    
    if range_match:
        start_date_str = range_match.group(1)
        start_year = range_match.group(2) or range_match.group(4)  # Use end year if start year not specified
        end_date_str = range_match.group(3)
        end_year = range_match.group(4)
        
        try:
            start_date = parser.parse(f"{start_date_str} {start_year}")
            end_date = parser.parse(f"{end_date_str} {end_year}")
            
            result["start_date"] = int(start_date.timestamp())
            result["end_date"] = int(end_date.timestamp())
            return result
        except Exception as e:
            print(f"Error parsing date range: {e}")
    
    # Case: Simple date range (e.g., "15 - 27 March 2025")
    simple_range_pattern = r'(\d+)\s*-\s*(\d+)\s+([A-Za-z]+)\s+(\d{4})'
    simple_range_match = re.search(simple_range_pattern, date_string)
    
    if simple_range_match:
        start_day = simple_range_match.group(1)
        end_day = simple_range_match.group(2)
        month = simple_range_match.group(3)
        year = simple_range_match.group(4)
        
        try:
            start_date = parser.parse(f"{start_day} {month} {year}")
            end_date = parser.parse(f"{end_day} {month} {year}")
            
            result["start_date"] = int(start_date.timestamp())
            result["end_date"] = int(end_date.timestamp())
            return result
        except Exception as e:
            print(f"Error parsing simple date range: {e}")
    
    # Case: Expiration dates (e.g., "Expires on 30 June 2025")
    expiry_pattern = r'Expires?\s+on\s+(\d+\s+[A-Za-z]+\s+\d{4})'
    expiry_match = re.search(expiry_pattern, date_string)
    
    if expiry_match:
        expiry_date_str = expiry_match.group(1)
        try:
            expiry_date = parser.parse(expiry_date_str)
            
            # For expiry dates, we'll set start as today and end as the expiry date
            result["start_date"] = int(today.timestamp())
            result["end_date"] = int(expiry_date.timestamp())
            return result
        except:
            pass

    # Case: Single date (e.g., "15 March 2025")
    single_date_pattern = r'(\d+)\s+([A-Za-z]+)\s+(\d{4})'
    single_date_match = re.search(single_date_pattern, date_string)
    
    if single_date_match:
        day = single_date_match.group(1)
        month = single_date_match.group(2)
        year = single_date_match.group(3)
        
        try:
            single_date = parser.parse(f"{day} {month} {year}")
            result["start_date"] = int(single_date.timestamp())
            result["end_date"] = int(single_date.timestamp())
            return result
        except Exception as e:
            print(f"Error parsing single date: {e}")
            
    return result