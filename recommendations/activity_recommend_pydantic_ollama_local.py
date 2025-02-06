from pydantic import BaseModel, Field, field_validator
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
    MANDARIN = "Mandarin"
    MALAY = "Malay"
    TAMIL = "Tamil"

class LanguageFluency(str, Enum):
    NONE = "None"
    BASIC = "Basic"
    FLUENT = "Fluent"
    NA = "NA"

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class Occupation(str, Enum):
    STUDENT = "Student"
    PART_TIME = "Part Time"
    FREELANCE = "Freelance"
    UNEMPLOYED = "Unemployed"
    EMPLOYED = "Employed"
    NA = "NA"

class AcademicQualification(str, Enum):
    PRIMARY_SCHOOL = "Primary School"
    PSLE = "PSLE"
    GCE_O = "GCE_O_LEVEL"
    GCE_N = "GCE_N_LEVEL"
    GCE_A = "GCE_A_LEVEL"
    NITEC = "NITEC"
    DIPLOMA = "Diploma"
    BACHELOR = "Bachelor's Degree"
    MASTER = "Master's Degree"
    DOCTORAL = "Doctoral Degree"
    GRADUATE_DIPLOMA = "Graduate Diploma"
    OTHERS = "Others"

class DrivingLicenseType(str, Enum):
    CLASS2 = "Class 2"
    CLASS3 = "Class 3"
    CLASS4 = "Class 4"
    CLASS5 = "Class 5"
    NA = "NA"

class TimeSlot(BaseModel):
    start_time: time
    end_time: time

    @field_validator('end_time')
    def end_time_must_be_after_start_time(cls, v: time, info) -> time:
        start_time = info.data.get('start_time')
        if start_time is not None and v <= start_time:
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
    driving_licenses_type: List[DrivingLicenseType]
    gender: Gender
    languages: List[Language]
    occupation: Occupation
    highest_academic_qualification: AcademicQualification
    english_language_proficiency: LanguageFluency
    mandarin_language_proficiency: LanguageFluency
    tamil_language_proficiency: LanguageFluency
    malay_language_proficiency: LanguageFluency
    experience_years: float = Field(ge=0)
    volunteering_experiences: str
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

    @field_validator('max_volunteers')
    def max_volunteers_must_be_greater_than_min(cls, v: int, info) -> int:
        min_volunteers = info.data.get('min_volunteers')
        if min_volunteers is not None and v < min_volunteers:
            raise ValueError('max_volunteers must be greater than or equal to min_volunteers')
        return v

    class Config:
        use_enum_values = True

class Match(BaseModel):
    score: float = Field(ge=0, le=1)
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
            ],
            "stream": False
        }
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            response_json = response.json()
            if not isinstance(response_json, dict) or 'message' not in response_json:
                raise ValueError("Unexpected response format from Ollama API")
            
            content = response_json['message']['content']
            
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx == -1 or end_idx <= 0:
                raise ValueError("No JSON object found in response")
            
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Transform the response to match the Pydantic model fields
            transformed_result = {
                "score": result.get("matching_score", 0.0),  # Convert matching_score to score
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "potential_challenges": result.get("potential_challenges", [])
            }
            
            # Ensure score is between 0 and 1
            transformed_result["score"] = max(0, min(1, transformed_result["score"]))
                
            return json.dumps(transformed_result)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from Ollama: {e}")
        except Exception as e:
            raise Exception(f"Error processing Ollama response: {e}")

    def analyze_match(self, volunteer: Volunteer, activity: Activity) -> Match:
        # Example of a must requirements therefore it is better to create 
        # validation by ourselves, even though LLM might recognize this issue
        # but it still depends on how the LLM interpret it
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
        Analyze this volunteer-activity match and return a JSON object with exactly this format:
        {{
            "matching_score": (number between 0-1),
            "reasoning": "detailed explanation",
            "potential_challenges": ["challenge1", "challenge2", ...]
        }}

        Volunteer:
        - Name: {volunteer.name}
        - Age: {volunteer.age}
        - Skills: {', '.join(volunteer.skills)}
        - Languages: {', '.join(volunteer.languages)}
        - Physical Capability: {volunteer.physical_capability}
        - Has Driving License: {volunteer.has_driving_license}
        - Experience: {volunteer.experience_years} years

        Activity:
        - Name: {activity.name}
        - Required Skills: {', '.join(activity.required_skills)}
        - Physical Requirement: {activity.required_physical_capability}
        - Required Languages: {', '.join(activity.required_languages)}
        - Requires Driving: {activity.requires_driving_license}
        - Minimum Experience: {activity.min_experience_years} years
        """

        try:
            response = self._call_ollama(prompt)
            result = json.loads(response)
            return Match(**result)
        except Exception as e:
            print(f"Error processing match: {e}")
            return Match(
                score=0.0,
                reasoning="Error analyzing match",
                potential_challenges=["Error processing response"]
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
            driving_licenses_type=[DrivingLicenseType.CLASS2],
            gender=Gender.FEMALE,
            english_language_proficiency=LanguageFluency.FLUENT,
            mandarin_language_proficiency=LanguageFluency.BASIC,
            tamil_language_proficiency=LanguageFluency.NONE,
            malay_language_proficiency=LanguageFluency.NONE,
            languages=[Language.ENGLISH, Language.MANDARIN],
            occupation=Occupation.EMPLOYED,
            highest_academic_qualification=AcademicQualification.BACHELOR,
            volunteering_experiences="",
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
            driving_licenses_type=[],
            gender=Gender.MALE,
            english_language_proficiency=LanguageFluency.FLUENT,
            mandarin_language_proficiency=LanguageFluency.FLUENT,
            tamil_language_proficiency=LanguageFluency.NONE,
            malay_language_proficiency=LanguageFluency.NONE,
            languages=[Language.ENGLISH, Language.MANDARIN],
            occupation=Occupation.STUDENT,
            highest_academic_qualification=AcademicQualification.DIPLOMA,
            volunteering_experiences="",
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
            required_languages=[Language.ENGLISH, Language.MANDARIN],
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