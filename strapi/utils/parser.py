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
    
    print(f"Current date: {today.strftime('%Y-%m-%d')} (timestamp: {int(today.timestamp())})")
    print(f"End of year: {end_of_year.strftime('%Y-%m-%d')} (timestamp: {int(end_of_year.timestamp())})")
    
    # Input validation
    if not date_string or not isinstance(date_string, str) or date_string.strip() == "":
        print(f"Using default dates due to invalid input: {date_string}")
        return result
    
    # Handle case: Ongoing events
    if date_string.lower() in ["now on till further notice", "ongoing"]:
        print(f"Handling ongoing event: {date_string}")
        return result
    
    print(f"Parsing date string: '{date_string}'")
    
    # Case: Specific days of the week in a month (e.g., "6, 13, 20 & 27 Mar 2025 (every Thursday)")
    specific_days_pattern = r'(\d+(?:,\s*\d+)*(?:\s*&\s*\d+)?)\s+([A-Za-z]+)\s+(\d{4})'
    specific_days_match = re.search(specific_days_pattern, date_string)
    
    if specific_days_match:
        days = re.findall(r'\d+', specific_days_match.group(1))
        month = specific_days_match.group(2)
        year = specific_days_match.group(3)
        
        print(f"Matched specific days pattern: days={days}, month={month}, year={year}")
        
        all_dates = []
        for day in days:
            try:
                date_str = f"{day} {month} {year}"
                date_obj = parser.parse(date_str)
                all_dates.append(date_obj)
                print(f"Parsed date: {date_str} → {date_obj.strftime('%Y-%m-%d')}")
            except Exception as e:
                print(f"Error parsing specific day '{day}': {e}")
                continue
        
        if all_dates:
            all_dates.sort()
            result["start_date"] = int(all_dates[0].timestamp())
            result["end_date"] = int(all_dates[-1].timestamp())
            print(f"Using specific days: start={all_dates[0].strftime('%Y-%m-%d')}, end={all_dates[-1].strftime('%Y-%m-%d')}")
            return result
    
    # Case: Date range with days of week (e.g., "17 Feb - 27 Mar 2025 (Monday-Thursday)")
    range_pattern = r'(\d+\s+[A-Za-z]+)(?:\s+(\d{4}))?\s*-\s*(\d+\s+[A-Za-z]+)\s+(\d{4})'
    range_match = re.search(range_pattern, date_string)
    
    if range_match:
        start_date_str = range_match.group(1)
        start_year = range_match.group(2) or range_match.group(4)  # Use end year if start year not specified
        end_date_str = range_match.group(3)
        end_year = range_match.group(4)
        
        print(f"Matched date range pattern: start='{start_date_str} {start_year}', end='{end_date_str} {end_year}'")
        
        try:
            start_date = parser.parse(f"{start_date_str} {start_year}")
            end_date = parser.parse(f"{end_date_str} {end_year}")
            
            # Check if dates are in the past, if so use current date as fallback
            if start_date < today:
                print(f"WARNING: Start date {start_date.strftime('%Y-%m-%d')} is in the past!")
                if int(start_year) < today.year:
                    print(f"Year mismatch detected: {start_year} vs current {today.year}")
                    # Try to correct the year while keeping month and day
                    start_date = start_date.replace(year=today.year)
                    end_date = end_date.replace(year=today.year)
                    print(f"Adjusted dates to current year: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
                    
                    # If still in the past, use today
                    if start_date < today:
                        print(f"Adjusted start date still in past, using today")
                        start_date = today
            
            result["start_date"] = int(start_date.timestamp())
            result["end_date"] = int(end_date.timestamp())
            print(f"Using date range: start={start_date.strftime('%Y-%m-%d')}, end={end_date.strftime('%Y-%m-%d')}")
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
        
        print(f"Matched simple date range: start_day={start_day}, end_day={end_day}, month={month}, year={year}")
        
        try:
            start_date = parser.parse(f"{start_day} {month} {year}")
            end_date = parser.parse(f"{end_day} {month} {year}")
            
            # Check if dates are in the past, if so use current date as fallback
            if start_date < today:
                print(f"WARNING: Start date {start_date.strftime('%Y-%m-%d')} is in the past!")
                if int(year) < today.year:
                    print(f"Year mismatch detected: {year} vs current {today.year}")
                    # Try to correct the year while keeping month and day
                    start_date = start_date.replace(year=today.year)
                    end_date = end_date.replace(year=today.year)
                    print(f"Adjusted dates to current year: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
                    
                    # If still in the past, use today
                    if start_date < today:
                        print(f"Adjusted start date still in past, using today")
                        start_date = today
            
            result["start_date"] = int(start_date.timestamp())
            result["end_date"] = int(end_date.timestamp())
            print(f"Using simple date range: start={start_date.strftime('%Y-%m-%d')}, end={end_date.strftime('%Y-%m-%d')}")
            return result
        except Exception as e:
            print(f"Error parsing simple date range: {e}")
    
    # Case: Expiration dates (e.g., "Expires on 30 June 2025")
    expiry_pattern = r'Expires?\s+on\s+(\d+\s+[A-Za-z]+\s+\d{4})'
    expiry_match = re.search(expiry_pattern, date_string)
    
    if expiry_match:
        expiry_date_str = expiry_match.group(1)
        print(f"Matched expiration date: {expiry_date_str}")
        
        try:
            expiry_date = parser.parse(expiry_date_str)
            
            # Check if date is in the past, if so use end of year as fallback
            if expiry_date < today:
                print(f"WARNING: Expiry date {expiry_date.strftime('%Y-%m-%d')} is in the past!")
                if expiry_date.year < today.year:
                    print(f"Year mismatch detected: {expiry_date.year} vs current {today.year}")
                    # Use end of current year
                    expiry_date = end_of_year
                    print(f"Using end of current year instead: {expiry_date.strftime('%Y-%m-%d')}")
            
            # For expiry dates, we'll set start as today and end as the expiry date
            result["start_date"] = int(today.timestamp())
            result["end_date"] = int(expiry_date.timestamp())
            print(f"Using expiry date range: start={today.strftime('%Y-%m-%d')}, end={expiry_date.strftime('%Y-%m-%d')}")
            return result
        except Exception as e:
            print(f"Error parsing expiry date: {e}")

    # Case: Single date (e.g., "15 March 2025")
    single_date_pattern = r'(\d+)\s+([A-Za-z]+)\s+(\d{4})'
    single_date_match = re.search(single_date_pattern, date_string)
    
    if single_date_match:
        day = single_date_match.group(1)
        month = single_date_match.group(2)
        year = single_date_match.group(3)
        
        print(f"Matched single date: day={day}, month={month}, year={year}")
        
        try:
            date_str = f"{day} {month} {year}"
            single_date = parser.parse(date_str)
            
            # Check if date is in the past, if so use today as fallback
            if single_date < today:
                print(f"WARNING: Single date {single_date.strftime('%Y-%m-%d')} is in the past!")
                if int(year) < today.year:
                    print(f"Year mismatch detected: {year} vs current {today.year}")
                    # Try to correct the year while keeping month and day
                    single_date = single_date.replace(year=today.year)
                    print(f"Adjusted date to current year: {single_date.strftime('%Y-%m-%d')}")
                    
                    # If still in the past, use today
                    if single_date < today:
                        print(f"Adjusted date still in past, using today")
                        single_date = today
            
            result["start_date"] = int(single_date.timestamp())
            result["end_date"] = int(single_date.timestamp())
            print(f"Using single date: {single_date.strftime('%Y-%m-%d')}")
            return result
        except Exception as e:
            print(f"Error parsing single date: {e}")
    
    print(f"No date patterns matched, using default dates: start={today.strftime('%Y-%m-%d')}, end={end_of_year.strftime('%Y-%m-%d')}")
    return result