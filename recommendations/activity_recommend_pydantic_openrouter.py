from pydantic import BaseModel, Field, field_validator
from typing import List, Set, Dict, Optional
from datetime import datetime, time
import json
import requests
from time import sleep
from enum import Enum
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

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
    def __init__(self, model_name: str = "gpt-4", api_key: str = None, base_url: str = "https://openrouter.ai/api/v1"):
        # Add validation for API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        self.model_name = model_name
        self.base_url = base_url
        
        # Initialize OpenAI client with required headers for OpenRouter
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": "http://localhost:5000",  # Required by OpenRouter
                "X-Title": "Volunteer Matching System"     # Required by OpenRouter
            }
        )

    def _call_openrouter(self, prompt: str) -> str:
        try:
            # Print debug information
            print(f"Calling OpenRouter API with model: {self.model_name}")
            print(f"Using base URL: {self.base_url}")
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert volunteer coordinator. Analyze volunteer-activity matches and provide a score between 0-1, 
                        detailed reasoning, and potential challenges. Respond ONLY with a JSON object in this exact format:
                        {
                            "score": (number between 0-1),
                            "reasoning": "detailed explanation string",
                            "potential_challenges": ["challenge1", "challenge2", ...]
                        }"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Validate completion object
            if not completion or not hasattr(completion, 'choices'):
                raise ValueError("Invalid response structure from API")
            
            if not completion.choices:
                raise ValueError("No choices in API response")
                
            message = completion.choices[0].message
            if not message or not message.content:
                raise ValueError("No content in API response")
            
            # Try to parse the response as JSON
            parsed_response = json.loads(message.content)
            
            # Transform the response if needed
            if "matching_score" in parsed_response:
                parsed_response["score"] = parsed_response.pop("matching_score")
                
            # Ensure all required fields are present
            required_fields = {"score", "reasoning", "potential_challenges"}
            if not all(field in parsed_response for field in required_fields):
                raise ValueError(f"Missing required fields. Response: {parsed_response}")
                
            return json.dumps(parsed_response)

        except Exception as e:
            print(f"API Error detail: {str(e)}")  # Debug print
            raise Exception(f"API Error: {str(e)}")

    def analyze_match(self, volunteer: Volunteer, activity: Activity) -> Match:
        # Pre-validation checks
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
        Please analyze this volunteer-activity match:

        Volunteer Details:
        - Name: {volunteer.name}
        - Age: {volunteer.age}
        - Skills: {', '.join(volunteer.skills)}
        - Languages: {', '.join(volunteer.languages)}
        - Physical Capability: {volunteer.physical_capability}
        - Has Driving License: {volunteer.has_driving_license}
        - Experience: {volunteer.experience_years} years

        Activity Requirements:
        - Name: {activity.name}
        - Required Skills: {', '.join(activity.required_skills)}
        - Physical Requirement: {activity.required_physical_capability}
        - Required Languages: {', '.join(activity.required_languages)}
        - Requires Driving: {activity.requires_driving_license}
        - Minimum Experience: {activity.min_experience_years} years

        Consider:
        1. Skill match
        2. Language requirements
        3. Physical capability compatibility
        4. Experience level
        5. Driving requirements

        Provide a score between 0 and 1, where 1 indicates a perfect match.
        """

        try:
            response = self._call_openrouter(prompt)
            result = json.loads(response)
            
            # Validate score range
            score = float(result["score"])
            if not 0 <= score <= 1:
                score = max(0, min(1, score))  # Clamp to [0,1]
                
            return Match(
                score=score,
                reasoning=result["reasoning"],
                potential_challenges=result["potential_challenges"]
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")  # For debugging
            return Match(
                score=0.0,
                reasoning="Error: Could not parse API response",
                potential_challenges=["API response format error"]
            )
        except Exception as e:
            print(f"General Error: {str(e)}")  # For debugging
            return Match(
                score=0.0,
                reasoning=f"Error analyzing match: {str(e)}",
                potential_challenges=["Error processing response"]
            )

# For testing purposes
def test_matcher():
    try:
        # Initialize matcher with debug information
        print("Initializing VolunteerMatcher...")
        matcher = VolunteerMatcher(
            model_name="meta-llama/llama-3.2-3b-instruct:free", # Try a different model
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Test with a simple prompt
        print("Testing API connection...")
        test_prompt = "Give a brief summary of volunteer matching importance."
        response = matcher._call_openrouter(test_prompt)
        print(f"Test response: {response}")
        
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

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
        matcher = VolunteerMatcher(model_name="meta-llama/llama-3.2-3b-instruct:free", api_key=os.getenv("OPENAI_API_KEY"), base_url="https://openrouter.ai/api/v1")
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