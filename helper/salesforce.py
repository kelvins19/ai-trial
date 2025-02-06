from simple_salesforce import Salesforce
from dotenv import load_dotenv
import os
import json


load_dotenv()
salesforce_username = os.environ.get("SALESFORCE_USER_NAME")
salesforce_password = os.environ.get("SALESFORCE_USER_PASSWORD")
salesforce_security_token = os.environ.get("SALESFORCE_USER_SECURITY_TOKEN")
salesforce_vms_org_id = os.environ.get("SALESFORCE_VMS_ORG_ID")

def get_salesforce_client():
    sf = Salesforce(username=salesforce_username, password=salesforce_password, security_token=salesforce_security_token)
    return sf

def get_salesforce_org_client():
    sf = Salesforce(password=salesforce_password, username=salesforce_username, organizationId=salesforce_vms_org_id)
    return sf

def get_volunteer_contacts(sf):
    contacts = sf.query("SELECT Id, FirstName, LastName, Professional_Skills__c, Interest_Skills__c, Volunteer_Interest__c, Spiritual__c, Outreach_Interest__c, Reason_For_Volunteering__c FROM Contact WHERE Volunteer__c = TRUE")

    # Convert to JSON
    contacts_json = json.loads(json.dumps(contacts))


    # Store JSON in a variable for further use
    contacts_data = contacts_json
    return contacts_data

def get_volunteer_contact_by_id(sf, id):
    contact = sf.Contact.get(id)

    # Convert to JSON
    contact_json = json.loads(json.dumps(contact))

    contact_data = contact_json
    return contact_data

sf = get_salesforce_client()
contacts = get_volunteer_contacts(sf)

# Print JSON with nice formatting
print(json.dumps(contacts["records"], indent=2))