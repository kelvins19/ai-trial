from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import requests
from time import sleep

@dataclass
class Volunteer:
    id: int
    description: str
    name: str

@dataclass
class Activity:
    id: int
    description: str
    name: str
    
@dataclass
class Match:
    score: float
    reasoning: str
    potential_challenges: List[str]

class OllamaVolunteerMatcher:
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._verify_ollama_connection()
        
    def _verify_ollama_connection(self):
        """Checking that Ollama is running and the model is available."""
        try:
            # Test the connection with a simple request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": "test"}
                    ]
                }
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Could not connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is installed and running."
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception(
                    f"Model '{self.model_name}' not found. "
                    f"Please run 'ollama pull {self.model_name}' to download it."
                )
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
            return response.json()['message']['content']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
            
    def analyze_match(self, volunteer: Volunteer, activity: Activity) -> Match:
        prompt = f"""
        Analyze how well this volunteer matches with the activity. Consider skills, requirements, and fit.
        Provide a matching score and explanation in JSON format.

        Volunteer Description:
        {volunteer.description}

        Activity Description:
        {activity.description}

        Return only a JSON object in this exact format:
        {{
            "matching_score": <number between 0 and 1>,
            "reasoning": "<detailed explanation>",
            "potential_challenges": ["<challenge1>", "<challenge2>", ...]
        }}
        """

        try:
            response = self._call_ollama(prompt)
            
            # Find the JSON object in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            return Match(
                score=float(result["matching_score"]),
                reasoning=result["reasoning"],
                potential_challenges=result["potential_challenges"]
            )
        except Exception as e:
            print(f"Error processing match: {e}")
            print(f"Raw response: {response}")
            return Match(
                score=0.0,
                reasoning="Error analyzing match",
                potential_challenges=["Error processing response"]
            )

    # Set min score 0.6 as the minimum requirement to show the matched activities
    def find_best_matches(self, volunteer: Volunteer, activities: List[Activity], 
                         min_score: float = 0.6) -> List[tuple[Activity, Match]]:
        matches = []
        
        for activity in activities:
            match = self.analyze_match(volunteer, activity)
            if match.score >= min_score:
                matches.append((activity, match))
            sleep(0.5)  # Small delay between requests
        
        matches.sort(key=lambda x: x[1].score, reverse=True)
        return matches
    
# Generate sample data for testing purposes
def generate_sample_data():
    volunteers = [
        Volunteer(
            id=1,
            name="Maria Rodriguez",
            description="""
            I am a bilingual (English/Spanish) retired nurse with 25 years of experience in pediatric care.
            I have a valid driver's license and own vehicle. I'm comfortable with computers and have experience
            with Zoom and Microsoft Office. I enjoy working with children and elderly people. I can lift up to
            20 pounds and can stand for moderate periods. Available Monday through Thursday mornings and early
            afternoons. Current CPR certification and clean background check.
            """
        ),
        Volunteer(
            id=2,
            name="James Chen",
            description="""
            College student majoring in Computer Science. Fluent in English and Mandarin. Strong technical
            skills including programming and digital content creation. No driver's license. Available weekends
            only. Comfortable with remote work and technology. No formal background check completed. Can't do
            heavy lifting due to back injury but can handle desk work and light activities.
            """
        ),
        Volunteer(
            id=3,
            name="Sarah Thompson",
            description="""
            Professional chef with 15 years of experience in restaurant management. Food safety certified.
            Strong leadership skills. Available evenings and weekends. Can lift heavy items (up to 50 pounds)
            and stand for long periods. Has driver's license and own car. Background check completed. 
            Speaks only English. Experienced in inventory management and team coordination.
            """
        ),
        Volunteer(
            id=4,
            name="Ahmed Hassan",
            description="""
            Semi-retired accountant with expertise in bookkeeping and financial management. Available
            weekday afternoons only. No physical labor due to heart condition. Prefers quiet, indoor
            environments. Has driver's license but prefers not to drive long distances. Speaks English
            and Arabic. Good with spreadsheets and financial software. Has background check.
            """
        ),
        Volunteer(
            id=5,
            name="Emily Watson",
            description="""
            High school student looking for weekend volunteer opportunities. Enthusiastic about helping
            others but has no professional experience. Good with social media and technology. No driver's
            license. No background check. Can only volunteer with parental consent as she's 16 years old.
            Plays volleyball and is physically active. Available Saturdays and Sundays only.
            """
        )
    ]
    
    activities = [
        Activity(
            id=1,
            name="School Health Room Assistant",
            description="""
            Elementary school needs volunteers to assist the school nurse. Duties include basic health
            screenings, medical records maintenance, and first aid for minor injuries. Medical background
            required. Background check and comfort with children required. Spanish language skills highly
            desired. Hours: Monday-Friday 9AM-1PM. Basic computer skills needed. Must have at least 5 years
            of healthcare experience.
            """
        ),
        Activity(
            id=2,
            name="Senior Center Health Education",
            description="""
            Need healthcare professionals for senior health education sessions. Topics: diabetes management,
            nutrition, and wellness. Sessions are seated. Computer skills needed for presentations.
            Available times: Monday and Wednesday mornings, 10AM-12PM. Must work well with elderly.
            Background check required. Healthcare background preferred.
            """
        ),
        Activity(
            id=3,
            name="Food Bank Warehouse Coordinator",
            description="""
            Managing food bank warehouse operations. Requires heavy lifting (40+ pounds), standing for long
            periods, and organizing inventory. Must have driver's license for occasional food pickup.
            Available shifts: Tuesday-Saturday 7AM-3PM. Experience with inventory management preferred.
            Background check required. Must be able to manage volunteer teams and handle food safely.
            """
        ),
        Activity(
            id=4,
            name="Virtual Tutoring Program",
            description="""
            Online tutoring for high school students in math and science. Must be comfortable with video
            conferencing and digital teaching tools. Available weekday evenings 4PM-8PM. College-level
            education in relevant subjects required. Background check needed. Must have reliable internet
            connection. Training provided.
            """
        ),
        Activity(
            id=5,
            name="Community Garden Assistant",
            description="""
            Help maintain community garden plots. Duties include planting, weeding, watering, and basic
            garden maintenance. Must be comfortable working outdoors in various weather conditions.
            Physical activity required. Available any day 8AM-6PM. No experience needed but gardening
            knowledge is a plus. Must be able to lift 20 pounds and use basic garden tools.
            """
        ),
        Activity(
            id=6,
            name="Hospital Reading Program",
            description="""
            Read to children in pediatric ward. Must have current flu shot and health screening. Background
            check and minimum age of 21 required. Available weekdays 2PM-4PM. Must be comfortable in hospital
            setting and working with sick children. Minimum one-year commitment required. Training provided.
            """
        ),
        Activity(
            id=7,
            name="Financial Literacy Workshop Leader",
            description="""
            Conduct workshops on basic financial management for community members. Must have background in
            finance, accounting, or related field. Available evenings 6PM-8PM. Must be comfortable public
            speaking and creating presentations. Background check required. Bilingual skills a plus.
            """
        ),
        Activity(
            id=8,
            name="Youth Sports Assistant",
            description="""
            Assist with after-school sports programs. Must be able to run, demonstrate exercises, and
            supervise physical activities. Available Monday-Friday 3PM-6PM. Must be at least 18 years old
            with background check. First aid certification required. Experience working with children needed.
            """
        )
    ]
    
    return volunteers, activities

def main():
    print("\nInitializing volunteer matcher...\n")
    
    try:
        matcher = OllamaVolunteerMatcher(model_name="llama3")
    except Exception as e:
        print(f"Setup error: {e}")
        return

    volunteers, activities = generate_sample_data()
    
     # Analyze matches for each volunteer
    for volunteer in volunteers:
        print(f"\n{'='*80}")
        print(f"Analyzing matches for volunteer {volunteer.id}: {volunteer.name}")
        print(f"{'='*80}")
        
        try:
            matches = matcher.find_best_matches(volunteer, activities, min_score=0.6)
            
            if not matches:
                print("\nNo suitable matches found above the minimum score threshold.")
                print("\nPossible reasons for no matches:")
                
                # Get a few low-scoring matches to explain why they didn't meet the threshold
                low_matches = matcher.find_best_matches(volunteer, activities, min_score=0.0)[:2]
                for activity, match in low_matches:
                    print(f"\nActivity: {activity.name}")
                    print(f"Score: {match.score:.2f}")
                    print("Challenges:")
                    for challenge in match.potential_challenges:
                        print(f"- {challenge}")
            else:
                print(f"\nFound {len(matches)} suitable matches:")
                for activity, match in matches:
                    print(f"\nActivity: {activity.name}")
                    print(f"Matching Score: {match.score:.2f}")
                    print("\nReasoning:")
                    print(match.reasoning)
                    print("\nPotential Challenges:")
                    for challenge in match.potential_challenges:
                        print(f"- {challenge}")
            
            print(f"\n{'-'*80}")
                
        except Exception as e:
            print(f"Error during matching process: {e}")

if __name__ == "__main__":
    main()