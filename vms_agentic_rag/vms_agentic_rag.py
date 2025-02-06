import json
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import nest_asyncio
from utils.timer import Timer
import asyncio
import time as time_module 
from pydantic import BaseModel, Field, field_validator
import os
from dotenv import load_dotenv
from memory_profiler import profile

load_dotenv()
nest_asyncio.apply()

class Deps(BaseModel):
    phone_number: str
    prompt: str

class VMSAgenticRag:
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None, prompt: str = None):
        # Use CONFIG constant for default values
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.model = OpenAIModel(
            model_name=self.model_name, 
            api_key=self.api_key,
            base_url=self.base_url,
        )

        system_prompt = "You are a support chatbot for volunteer management system."
        if prompt != "": 
            system_prompt = prompt

        system_prompt += """Where user are going to send the datas and 
        you have to summarize the detail in user friendly format, that can be easily understood by users.
        Store the session for 1 hour, so that user can continue the session later and 
        we can retrieve the message history based on the user session.
        Do not return the session id, only the summarize details.

        If the prompt is like this: Cancel no {{the listed number}}, return the response in a json only like this:
        {{
            "activity_id": activity.ID,
            "volunteer_activity_id": volunteer_activity_id,
            "action": "cancel_activity",
            "phone_number": Session ID
        }}
        Make sure to check if the activity.ID is exist within the list of activities in previous session, 
        If not exist, return appropriate error response.

        If the user inputs a number only:
        1 - Execute get_list_activities tool
        2 - Execute get_my_schedule tool
        3 - Execute get_list_activities_to_cancel tool
        Other Number - Execute get_action_menu and also return message that states that input is not supported.

        If the user inputs a prompt that is not recognized at all, execute get_action_menu.
        """

        # print(f"Current system prompt: {system_prompt}")

        self.agent = Agent(self.model, system_prompt=system_prompt, deps_type=Deps)
        self.timer = Timer()  # Add this line

        @self.agent.tool  
        def get_list_activities(ctx: RunContext[str]) -> str:
            """Get list of activities"""
            phone_number = ctx.deps.phone_number
            print(f"Phone number: {phone_number}")

            try:
                # print("Opening activities.json")
                with open('activities.json', 'r') as file:
                    activities_data = json.load(file)
                # print("Loaded activities.json")

                return f"List of activities {json.dumps(activities_data, indent=2)}" 
            except Exception as e:
                print(f"Exception: {str(e)}")
                return f"Error retrieving schedule: {str(e)}"

        @self.agent.tool  
        def get_my_schedule(ctx: RunContext[str]) -> str:
            """Get schedule for the volunteer"""
            phone_number = ctx.deps.phone_number
            print(f"Phone number: {phone_number}")
            try:
                # print("Opening volunteers.json")
                with open('volunteers.json', 'r') as file:
                    volunteers_data = json.load(file)
                # print("Loaded volunteers.json")
                
                # print("Opening volunteer_activities.json")
                with open('volunteer_activities.json', 'r') as file:
                    volunteer_activities_data = json.load(file)
                # print("Loaded volunteer_activities.json")
                
                # print("Opening activities.json")
                with open('activities.json', 'r') as file:
                    activities_data = json.load(file)
                # print("Loaded activities.json")
                
                # Find the volunteer by phone number
                # print("Searching for volunteer")
                volunteer = next((v for v in volunteers_data if v['phone_number'] == phone_number), None)
                print(f"Volunteer: {volunteer}")
                
                if not volunteer:
                    return "Volunteer not found."
                
                volunteer_id = volunteer['id']
                print(f"Volunteer ID: {volunteer_id}")
                volunteer_activities = [activity for activity in volunteer_activities_data if activity['volunteer_id'] == volunteer_id]
                # print(f"Volunteer activities: {volunteer_activities}")
                
                if not volunteer_activities:
                    return "No activities found for this volunteer."
                
                # Get detailed activity data
                detailed_activities = []
                for volunteer_activity in volunteer_activities:
                    activity_id = volunteer_activity['activity_id']
                    activity_details = next((a for a in activities_data if a['id'] == activity_id), None)
                    if activity_details:
                        detailed_activity = {
                            "volunteer_activity_id": volunteer_activity['id'],  # Use the unique id from volunteer_activities
                            **volunteer_activity,
                            **activity_details
                        }
                        # Remove the conflicting id from activity_details
                        detailed_activity.pop('id', None)
                        detailed_activities.append(detailed_activity)
                
                return json.dumps(detailed_activities, indent=2)
            except Exception as e:
                print(f"Exception: {str(e)}")
                return f"Error retrieving schedule: {str(e)}"
            
        
        @self.agent.tool
        def get_list_activities_to_cancel(ctx: RunContext[str]) -> str:
            """Retrieving list for volunteer activities to cancel"""
            phone_number = ctx.deps.phone_number
            print(f"Phone number: {phone_number}")

            activities = get_my_schedule(ctx)
            # print(f"Activity details: {activities}")

            prompt = f"""
                Which activity would you like to cancel? 

                {activities} 

                Reply like this to cancel your activity: **Cancel no {{the listed number}}** e.g. Cancel no 1
                """

            return prompt
        
        @self.agent.tool
        def get_action_menu(ctx: RunContext[str]) -> str:
            """Retrieving action menu"""
            phone_number = ctx.deps.phone_number
            print(f"Phone number: {phone_number}")

            prompt = """
            Here are the list of actions you can perform:
            1. Get list of activities
            2. Get my schedule
            3. Get list of activities to cancel"""

            return prompt

    async def _call_openrouter_batch(self, deps: Deps) -> str:
        max_retries = 3
        retry_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                # Print debug information
                print(f"Calling OpenRouter API with model: {self.model_name}")
                print(f"Using base URL: {self.base_url}")

                print(f"User prompt: {deps.prompt}")
                self.timer.start()  # Start timer
                result = await self.agent.run(user_prompt=deps.prompt, deps=deps)
                self.timer.stop()  # Stop timer

                if result is None:
                    raise Exception("API returned no result")

                usage = result.usage()
                if usage is None:
                    raise Exception("API usage information is missing")

                total_tokens = usage.total_tokens
                total_time = self.timer.get_elapsed_time()

                print(f"Total tokens used: {total_tokens}")
                print(f"Total time taken: {total_time:.3f} ms")

                response = result.data

                # Log the response
                # print(f"API Response: {response}")

                if response.endswith("```}"):
                    response = response.replace("```}", "").strip()
                if response.endswith("```"):
                    response = response.replace("```", "").strip()
                if response.startswith("```"):
                    response = response.replace("```", "").strip()
                if response.startswith("json"):
                    response = response.replace("json", "").strip()

                return response

            except Exception as e:
                print(f"API Error detail: {str(e)}")  # Debug print
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time_module.sleep(retry_delay)  # Use the renamed import
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise Exception(f"API Error: {str(e)}")

    def _cancel_activity(self, phone_number: str, activity_id: int) -> str:
        try:
            with open('volunteers.json', 'r') as file:
                volunteers_data = json.load(file)
            
            with open('volunteer_activities.json', 'r') as file:
                volunteer_activities_data = json.load(file)
            
            # Find the volunteer by phone number
            volunteer = next((v for v in volunteers_data if v['phone_number'] == phone_number), None)
            
            if not volunteer:
                return json.dumps({"error": "Volunteer not found."})
            
            volunteer_id = volunteer['id']
            
            # Find the volunteer activity and change its status to "cancelled"
            for activity in volunteer_activities_data:
                if activity['activity_id'] == activity_id and activity['volunteer_id'] == volunteer_id:
                    activity['status'] = "cancelled"
                    break
            else:
                return json.dumps({"error": f"Activity ID {activity_id} does not exist in your list of activities to cancel."})
            
            # Save the updated volunteer_activities.json
            with open('volunteer_activities.json', 'w') as file:
                json.dump(volunteer_activities_data, file, indent=2)
            
            return f"Activity {activity_id} is successfully cancelled"
        except Exception as e:
            return f"Error: {str(e)}"

    async def get_response(self, input_prompt: str, phone_number: str) -> str:
        if input_prompt == "":
            input_prompt = "get_action_menu"
        
        prompt = f"""
        Session: {phone_number}

        {input_prompt}
        """

        vmsDeps = Deps(
            phone_number=phone_number,
            prompt=prompt
        )

        response = await self._call_openrouter_batch(vmsDeps)
        reply = response
        
        try:
            reply_json = json.loads(reply)
            print(f"Reply JSON: {reply_json}")
            if isinstance(reply_json, dict) and "activity_id" in reply_json:
                action = reply_json["action"]
                activity_id = reply_json["activity_id"]

                if action == "cancel_activity":
                    return self._cancel_activity(phone_number, activity_id)

            elif isinstance(reply_json, dict) and "error" in reply_json:
                return reply_json["error"]
        except json.JSONDecodeError:
            pass
        return reply