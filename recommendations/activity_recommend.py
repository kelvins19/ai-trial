from dataclasses import dataclass
from typing import List, Dict, Set
from datetime import datetime, time
import random

@dataclass
class Volunteer:
    id: int
    name: str
    age: int
    skills: Set[str]
    availability: Dict[str, List[time]]  # Day of week -> available times
    preferred_locations: List[str]
    has_driving_license: bool
    languages: List[str]
    experience_years: float
    background_check: bool
    physical_capability: str  # 'light', 'moderate', 'heavy'

@dataclass
class Activity:
    id: int
    name: str
    description: str
    required_skills: Set[str]
    location: str
    duration_hours: float
    min_volunteers: int
    max_volunteers: int
    required_physical_capability: str
    requires_driving_license: bool
    required_languages: List[str]
    requires_background_check: bool

class VolunteerMatcher:
    def __init__(self):
        self.volunteers = []
        self.activities = []

    def add_volunteer(self, volunteer: Volunteer):
        self.volunteers.append(volunteer)

    def add_activity(self, activity: Activity):
        self.activities.append(activity)

    def find_matches_for_volunteer(self, volunteer_id: int) -> List[Activity]:
        volunteer = next((v for v in self.volunteers if v.id == volunteer_id), None)
        if not volunteer:
            return []

        matches = []
        for activity in self.activities:
            if self._is_match(volunteer, activity):
                matches.append(activity)
        return matches

    def find_volunteers_for_activity(self, activity_id: int) -> List[Volunteer]:
        activity = next((a for a in self.activities if a.id == activity_id), None)
        if not activity:
            return []

        matches = []
        for volunteer in self.volunteers:
            if self._is_match(volunteer, activity):
                matches.append(volunteer)
        return matches

    def _is_match(self, volunteer: Volunteer, activity: Activity) -> bool:
        # Check all matching criteria
        if not volunteer.skills.issuperset(activity.required_skills):
            return False
        
        if activity.requires_driving_license and not volunteer.has_driving_license:
            return False
            
        if activity.requires_background_check and not volunteer.background_check:
            return False
            
        if not set(activity.required_languages).issubset(set(volunteer.languages)):
            return False
            
        physical_capability_levels = {'light': 1, 'moderate': 2, 'heavy': 3}
        if physical_capability_levels[volunteer.physical_capability] < \
           physical_capability_levels[activity.required_physical_capability]:
            return False
            
        if activity.location not in volunteer.preferred_locations:
            return False
            
        return True

def generate_sample_data(num_volunteers: int = 20, num_activities: int = 10) -> VolunteerMatcher:
    # Sample data pools
    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack",
             "Kelly", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Taylor"]
    
    all_skills = {"first_aid", "teaching", "cooking", "gardening", "counseling", "event_planning",
                 "social_media", "photography", "music", "sports_coaching", "art", "carpentry"}
    
    locations = ["Downtown", "North District", "South Side", "West End", "East Village", 
                "Central Park", "Harbor Area", "University District"]
    
    languages = ["English", "Spanish", "Mandarin", "French", "Arabic", "Hindi"]
    
    activity_names = [
        "Food Bank Distribution", "Senior Home Visit", "Youth Mentoring", "Park Cleanup",
        "Community Garden", "Homeless Shelter Support", "Reading to Children", "Sports Coaching",
        "Art Workshop", "Tech Support for Seniors"
    ]

    # Generate sample volunteers
    volunteers = []
    for i in range(num_volunteers):
        volunteers.append(Volunteer(
            id=i + 1,
            name=random.choice(names),
            age=random.randint(18, 70),
            skills=set(random.sample(list(all_skills), random.randint(2, 6))),
            availability={
                'Monday': [time(hour=h) for h in random.sample(range(8, 20), 3)],
                'Wednesday': [time(hour=h) for h in random.sample(range(8, 20), 3)],
                'Saturday': [time(hour=h) for h in random.sample(range(8, 20), 3)]
            },
            preferred_locations=random.sample(locations, random.randint(2, 4)),
            has_driving_license=random.choice([True, False]),
            languages=random.sample(languages, random.randint(1, 3)),
            experience_years=random.uniform(0, 10),
            background_check=random.choice([True, False]),
            physical_capability=random.choice(['light', 'moderate', 'heavy'])
        ))

    # Generate sample activities
    activities = []
    for i in range(num_activities):
        activities.append(Activity(
            id=i + 1,
            name=activity_names[i],
            description=f"Description for {activity_names[i]}",
            required_skills=set(random.sample(list(all_skills), random.randint(1, 3))),
            location=random.choice(locations),
            duration_hours=random.choice([2, 3, 4, 6, 8]),
            min_volunteers=random.randint(2, 5),
            max_volunteers=random.randint(6, 15),
            required_physical_capability=random.choice(['light', 'moderate', 'heavy']),
            requires_driving_license=random.choice([True, False]),
            required_languages=random.sample(languages, random.randint(1, 2)),
            requires_background_check=random.choice([True, False])
        ))

    # Create and populate the matcher
    matcher = VolunteerMatcher()
    for volunteer in volunteers:
        matcher.add_volunteer(volunteer)
    for activity in activities:
        matcher.add_activity(activity)

    return matcher

# Example usage
def main():
    # Generate sample data
    matcher = generate_sample_data()
    
    # Print all volunteers and their matches
    for volunteer in matcher.volunteers:
        print(f"\nVolunteer: {volunteer.name} (ID: {volunteer.id})")
        print(f"Skills: {', '.join(volunteer.skills)}")
        print(f"Languages: {', '.join(volunteer.languages)}")
        print(f"Physical Capability: {volunteer.physical_capability}")
        print(f"Has Driving License: {volunteer.has_driving_license}")
        print(f"Background Check: {volunteer.background_check}")
        
        matches = matcher.find_matches_for_volunteer(volunteer.id)
        print("\nMatching Activities:")
        for activity in matches:
            print(f"- {activity.name} ({activity.location})")
        print("-" * 50)

    # Print all activities and their matching volunteers
    for activity in matcher.activities:
        print(f"\nActivity: {activity.name}")
        print(f"Location: {activity.location}")
        print(f"Required Skills: {', '.join(activity.required_skills)}")
        print(f"Required Languages: {', '.join(activity.required_languages)}")
        print(f"Physical Capability Required: {activity.required_physical_capability}")
        
        matches = matcher.find_volunteers_for_activity(activity.id)
        print("\nMatching Volunteers:")
        for volunteer in matches:
            print(f"- {volunteer.name} (ID: {volunteer.id})")
        print("-" * 50)

if __name__ == "__main__":
    main()