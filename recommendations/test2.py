from pydantic import BaseModel, Field, validator
from typing import List, Set, Dict, Optional
from datetime import datetime, time
import json
import requests
from time import sleep
from enum import Enum

# Enums for constrained fields
class PhysicalCapability(str, Enum):
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"

class DayOfWeek(str, Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"

class Language(str, Enum):
    ENGLISH = "English"
    SPANISH = "Spanish"
    MANDARIN = "Mandarin"
    FRENCH = "French"
    ARABIC = "Arabic"
    HINDI = "Hindi"

class TimeSlot(BaseModel):
    start_time: time
    end_time: time

    @validator('end_time')
    def end_time_must_be_after_start_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v

class Availability(BaseModel):
    days: List[DayOfWeek]
    time_slots: List[TimeSlot]

class Volunteer(BaseModel):
    id: int
    name: str
    age: int = Field(ge=14)  # Minimum age of 14 for volunteers
    skills: Set[str]
    availability: Availability
    preferred_locations: List[str]
    has_driving_license: bool
    languages: List[Language]
    experience_years: float = Field(ge=0)
    background_check: bool
    physical_capability: PhysicalCapability
    description: str = Field(min_length=10)  # Ensure meaningful descriptions

    class Config:
        use_enum_values = True

class Activity(BaseModel):
    id: int
    name: str
    description: str = Field(min_length=10)
    required_skills: Set[str]
    location: str
    duration_hours: float = Field(gt=0)
    min_volunteers: int = Field(ge=1)
    max_volunteers: int = Field(ge=1)
    required_physical_capability: PhysicalCapability
    requires_driving_license: bool
    required_languages: List[Language]
    requires_background_check: bool
    schedule: Availability
    min_age: Optional[int] = Field(default=14, ge=14)
    min_experience_years: float = Field(default=0, ge=0)

    @validator('max_volunteers')
    def max_volunteers_must_be_greater_than_min(cls, v, values):
        if 'min_volunteers' in values and v < values['min_volunteers']:
            raise ValueError('max_volunteers must be greater than or equal to min_volunteers')
        return v

    class Config:
        use_enum_values = True

class Match(BaseModel):
    score: float = Field(ge=0, le=1)  # Rename field for consistency
    reasoning: str
    potential_challenges: List[str]

class VolunteerMatcher:
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "test"}]
                }
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Could not connect to Ollama at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception(f"Model '{self.model_name}' not found")
            raise Exception(f"Error connecting to Ollama: {e}")

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert volunteer coordinator. Analyze matches and provide responses in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        try:
            response = requests.post(url, json=data, stream=True)
            response.raise_for_status()

            combined_content = ""
            for line in response.iter_lines():
                if line:  # Skip empty lines
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if "message" in chunk and "content" in chunk["message"]:
                            combined_content += chunk["message"]["content"]
                    except json.JSONDecodeError as e:
                        print(f"Error decoding chunk: {e}")
                        continue

            # Extract the first valid JSON object
            json_start = combined_content.find("{")
            json_end = combined_content.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON object found in response")
            
            json_response = combined_content[json_start:json_end]
            return json_response
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")




    def analyze_match(self, volunteer: Volunteer, activity: Activity) -> Match:
        if activity.min_age > volunteer.age:
            return Match(
                score=0.0,
                reasoning="Volunteer does not meet minimum age requirement",
                potential_challenges=["Age requirement not met"]
            )

        if activity.requires_background_check and not volunteer.background_check:
            return Match(
                score=0.0,
                reasoning="Activity requires background check",
                potential_challenges=["Missing background check"]
            )

        prompt = f"""
        Analyze how well this volunteer matches with the activity...
        """

        try:
            json_response = self._call_ollama(prompt)
            result = json.loads(json_response)
            return Match(**result)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return Match(
                score=0.0,
                reasoning="Error analyzing match",
                potential_challenges=["Failed to parse JSON response"]
            )
        except Exception as e:
            print(f"Error processing match: {e}")
            return Match(
                score=0.0,
                reasoning="Error analyzing match",
                potential_challenges=["Unexpected error"]
            )

def generate_sample_data():
    """Generate sample data using Pydantic models"""
    
    # Sample morning time slots
    morning_slots = [
        TimeSlot(start_time=time(9, 0), end_time=time(12, 0))
    ]
    
    # Sample afternoon time slots
    afternoon_slots = [
        TimeSlot(start_time=time(13, 0), end_time=time(17, 0))
    ]

    volunteers = [
        Volunteer(
            id=1,
            name="Maria Rodriguez",
            age=45,
            skills={"nursing", "first_aid", "patient_care"},
            availability=Availability(
                days=[DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY],
                time_slots=morning_slots
            ),
            preferred_locations=["Downtown", "North District"],
            has_driving_license=True,
            languages=[Language.ENGLISH, Language.SPANISH],
            experience_years=20.0,
            background_check=True,
            physical_capability=PhysicalCapability.MODERATE,
            description="Retired nurse with pediatric care experience."
        ),
        Volunteer(
            id=2,
            name="James Chen",
            age=20,
            skills={"programming", "tutoring", "digital_media"},
            availability=Availability(
                days=[DayOfWeek.SATURDAY, DayOfWeek.SUNDAY],
                time_slots=afternoon_slots
            ),
            preferred_locations=["University District"],
            has_driving_license=False,
            languages=[Language.ENGLISH, Language.MANDARIN],
            experience_years=1.0,
            background_check=False,
            physical_capability=PhysicalCapability.LIGHT,
            description="Computer Science student with tutoring experience."
        )
    ]

    activities = [
        Activity(
            id=1,
            name="School Health Room Assistant",
            description="Help school nurse with basic health screenings and first aid.",
            required_skills={"nursing", "first_aid"},
            location="Downtown",
            duration_hours=4.0,
            min_volunteers=1,
            max_volunteers=2,
            required_physical_capability=PhysicalCapability.MODERATE,
            requires_driving_license=False,
            required_languages=[Language.ENGLISH, Language.SPANISH],
            requires_background_check=True,
            schedule=Availability(
                days=[DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY],
                time_slots=morning_slots
            ),
            min_age=21,
            min_experience_years=2.0
        ),
        Activity(
            id=2,
            name="Virtual Tutoring",
            description="Online tutoring for high school students.",
            required_skills={"tutoring"},
            location="Remote",
            duration_hours=2.0,
            min_volunteers=1,
            max_volunteers=5,
            required_physical_capability=PhysicalCapability.LIGHT,
            requires_driving_license=False,
            required_languages=[Language.ENGLISH],
            requires_background_check=True,
            schedule=Availability(
                days=[DayOfWeek.SATURDAY],
                time_slots=afternoon_slots
            ),
            min_age=18,
            min_experience_years=0.0
        )
    ]

    return volunteers, activities

def main():
    try:
        matcher = VolunteerMatcher(model_name="llama3")
        volunteers, activities = generate_sample_data()
        
        print("\nAnalyzing volunteer matches...")
        for volunteer in volunteers:
            print(f"\n{'='*80}")
            print(f"Volunteer: {volunteer.name}")
            print(f"Skills: {', '.join(volunteer.skills)}")
            print(f"Languages: {', '.join(volunteer.languages)}")
            
            for activity in activities:
                match = matcher.analyze_match(volunteer, activity)
                print(f"\nActivity: {activity.name}")
                print(f"Match Score: {match.score:.2f}")
                print(f"Reasoning: {match.reasoning}")
                if match.potential_challenges:
                    print("Challenges:")
                    for challenge in match.potential_challenges:
                        print(f"- {challenge}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()