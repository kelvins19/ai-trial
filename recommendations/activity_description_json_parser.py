from dataclasses import dataclass
from typing import List, Set, Dict
import re

@dataclass
class ParsedRequirements:
    skills: Set[str]
    physical_capability: str
    driving_license_required: bool
    background_check_required: bool
    languages: Set[str]
    min_age: int
    time_commitment: str

class ActivityRequirementParser:
    def __init__(self):
        # Define keyword mappings for skills
        self.skill_keywords = {
            'first aid': 'first_aid',
            'cpr': 'first_aid',
            'teaching': 'teaching',
            'mentoring': 'teaching',
            'tutoring': 'teaching',
            'cooking': 'cooking',
            'food preparation': 'cooking',
            'gardening': 'gardening',
            'landscaping': 'gardening',
            'counseling': 'counseling',
            'mental health': 'counseling',
            'social media': 'social_media',
            'photography': 'photography',
            'music': 'music',
            'instrument': 'music',
            'coaching': 'sports_coaching',
            'athletic': 'sports_coaching',
            'carpentry': 'carpentry',
            'woodworking': 'carpentry'
        }

        # Physical capability keywords
        self.physical_keywords = {
            'light': ['light physical work', 'sitting', 'desk work', 'indoor activities'],
            'moderate': ['walking', 'standing', 'moderate physical', 'lifting up to 20 pounds'],
            'heavy': ['heavy lifting', 'construction', 'moving furniture', 'physically demanding']
        }

        # Language keywords
        self.language_keywords = {
            'english': ['english', 'english-speaking'],
            'spanish': ['spanish', 'español', 'spanish-speaking'],
            'mandarin': ['mandarin', 'chinese', 'mandarin-speaking'],
            'french': ['french', 'français', 'french-speaking'],
            'arabic': ['arabic', 'arabic-speaking'],
            'hindi': ['hindi', 'hindi-speaking']
        }

    def parse_requirements(self, description: str) -> ParsedRequirements:
        description = description.lower()
        
        # Initialize requirements
        requirements = ParsedRequirements(
            skills=set(),
            physical_capability='light',  # default to light
            driving_license_required=False,
            background_check_required=False,
            languages=set(),
            min_age=18,  # default minimum age
            time_commitment=""
        )

        # Extract skills
        for keyword, skill in self.skill_keywords.items():
            if keyword in description:
                requirements.skills.add(skill)

        # Determine physical capability
        for level, keywords in self.physical_keywords.items():
            if any(keyword in description for keyword in keywords):
                requirements.physical_capability = level
                break

        # Check for driving license requirement
        driving_keywords = ['driver', 'driving', 'license', 'car', 'transport', 'vehicle']
        if any(keyword in description for keyword in driving_keywords):
            requirements.driving_license_required = True

        # Check for background check requirement
        background_keywords = ['background check', 'background screening', 'police check', 'clearance']
        if any(keyword in description for keyword in background_keywords):
            requirements.background_check_required = True

        # Extract languages
        for language, keywords in self.language_keywords.items():
            if any(keyword in description for keyword in keywords):
                requirements.languages.add(language)

        # Extract minimum age
        age_pattern = r'(?:minimum|min|at least)\s+(\d+)\s+(?:years|year|yo|y\.o\.|years old)'
        age_match = re.search(age_pattern, description)
        if age_match:
            requirements.min_age = int(age_match.group(1))

        # Extract time commitment
        time_patterns = [
            r'(\d+)\s*hours? per (?:day|week|month)',
            r'(\d+)\s*hours? commitment',
            r'(\d+)-hour commitment'
        ]
        for pattern in time_patterns:
            time_match = re.search(pattern, description)
            if time_match:
                requirements.time_commitment = time_match.group(0)
                break

        return requirements

def main():
    # Example usage with different activity descriptions
    example_activities = [
        """
        Food Bank Distribution Assistant
        We need volunteers to help distribute food packages to local families.
        Requirements:
        - Must be able to lift up to 30 pounds
        - Driver's license required for food delivery
        - 4 hours per week commitment
        - Background check required
        - Spanish speaking preferred
        """,
        
        """
        Senior Center Activity Coordinator
        Looking for volunteers to organize activities for seniors.
        - Light physical work, mainly indoor activities
        - First aid certification is a plus
        - Must be at least 21 years old
        - English and Mandarin speaking required
        - 3 hours per day, twice a week
        """,
        
        """
        Community Garden Mentor
        Help maintain our community garden and teach gardening skills.
        Requirements:
        - Experience in gardening and teaching
        - Moderate physical work required
        - 5 hours per week commitment
        - Must be comfortable working outdoors
        - Background check required for working with youth
        """
    ]

    parser = ActivityRequirementParser()
    
    for i, description in enumerate(example_activities, 1):
        print(f"\nParsing Activity {i}:")
        print("-" * 50)
        print("Original Description:")
        print(description.strip())
        print("\nExtracted Requirements:")
        requirements = parser.parse_requirements(description)
        print(f"Skills: {requirements.skills}")
        print(f"Physical Capability: {requirements.physical_capability}")
        print(f"Driving License Required: {requirements.driving_license_required}")
        print(f"Background Check Required: {requirements.background_check_required}")
        print(f"Languages: {requirements.languages}")
        print(f"Minimum Age: {requirements.min_age}")
        print(f"Time Commitment: {requirements.time_commitment}")
        print("-" * 50)

if __name__ == "__main__":
    main()