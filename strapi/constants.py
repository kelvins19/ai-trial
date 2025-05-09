BASE_URL = "https://cms.bistrobytes.com.sg/api"
PATHS = {
    # "article": "/articles",
    "event": "/events",
    # "store": "/stores",
    "deal": "/deals",
    # "reward": "/rewards",
    # "config-contact-us": '/config-contact-us',
    # "config-about-us": '/config-about-us',
    # "config-reward": '/config-reward',
    # "config-kiosk": '/config-kiosk',
    # "config-content-footer": '/config-content-footer',
}

STAGING_INDEX_NAME = "i12katong-strapi"
LOCAL_INDEX_NAME = "i12katong-strapi-json"

STRAPI_KEYWORD_GENERATOR_PROMPT_1 = """
        You are an assistant tasked with identifying contents from query that is given. generate up to 10 keywords base on extracted information..
        I have an query json that may containing a descriptions and details of the given datas. Your task is to:
        Translate it into english if the given query text is not in english.
        Identify and extract the relevant details that can be readable by human from the query.
        Also if the details have abbreviations in there, also generate some relevant keywords from that abbreviations.
        Generate lowercase keywords based on the extracted information in multiple formats, including but not limited to:
        space separated, strip-separated, underscore_separated, no separation
        Ensure the keywords are diverse but still relevant to the extracted information.
        Also consider the category given from the details.
        Return the output in JSON format as follows:
        {{
            "keywords": [
                "<keyword1>",
                "<keyword2>",
                ...
            ]
        }}
        Example Output:
        If the extracted information is:
        Category: deals
        {{"data": {{"0": {{"body": "<p><strong><u>Weekday Specials F&B (Monday \u2013 Thursday)</u></strong></p><p><strong>FREE Movie Ticket</strong><br>Spend min. $80* and redeem a Golden Village movie ticket (worth $11.50)!</p><p><strong>PLUS! BONUS 1,000 Rewards+ Points</strong><br>When you spend at least $10 at participating F&amp;B shops on B1 as part of your $80* total spend.</p><p>List of participating B1 F&amp;B shops:</p><ul><li>Boost Juice</li><li>Edith Patisserie</li><li>Fun Toast</li><li>GOPIZZA</li><li>Kei Kaisendon</li><li>Maki-san</li><li>Ninja Mama</li><li>Pancake King &amp; Kopi</li><li>Pita Tree Kebabs</li><li>Rollgaadi</li><li>SG Hawker</li><li>Tea Pulse</li><li>The Fish &amp; Chips Shop</li><li>Toriten</li><li>Typhoon Caf\u00e9</li><li>Yum Yum Thai</li></ul><p><strong>SAFRA Exclusive</strong><br>Present your SAFRA membership card and redeem a FREE i12 Katong umbrella with a min. spend of $80*.</p><p>*Max. of 3 same-day receipts. Valid from Monday - Thursday only. Double spending required for supermarket and enrichment centres. Limited to the first 500 redemptions. Limited to 1 redemption per member per day. Redemptions must be made at the Concierge at Level 3. Other <a href=\\"https://shorturl.at/ixHVL\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">Terms &amp; Conditions </a>apply.</p>"
        The JSON output should be:
        {{
            "keywords": [
                "deals",
                "deal",
                "promo",
                "promos",
                "promotion",
                "promotions",
                "weekday specials",
                "events",
                "weekday",
                "deals",
                "bonus",
                "specials",
                "lc 2030cnt",
                "food and beverages",
                "food and beverage",
                "food&beverages",
                "food&beverage",
                "food & beverage",
                "food & beverages",
                "fnb",
                "f&b"
            ]
        }}
        Make sure the keywords are properly formatted and case-sensitive to match potential search queries. Make sure only return json
        """

STRAPI_SUMMARY_GENERATOR_PROMPT = """You are an assistant tasked with identifying contents from query that is given. 
        Generate summary of the given queries in human readable format with maximum up to 500 characters. 
        Disregard all informations that is too techical such as created and updated at data, dimensions and url of the images, """

STRAPI_DATE_GENERATOR_PROMPT = """
You are an assistant tasked with identifying the start and end dates from an event date string. 
Your task is to parse the given event date string and generate the start_date and end_date as Unix timestamps.
If the start_date cannot be detected, use today's date. If the end_date cannot be detected, use the end of the year date.
Return the output in JSON format as follows:
{
    "start_date": UNIX,
    "end_date": UNIX
}
Ensure the timestamps are accurate and represent the correct start and end dates of the event.
Example Input: "17 Feb - 27 Mar 2025 (Monday-Thursday)"
Example Output:
{
    "start_date": 1734566400,
    "end_date": 1740393600
}
"""

STRAPI_SUMMARY_AND_DATE_GENERATOR_PROMPT = """
You are an assistant tasked with identifying contents from a given query. 
Your task is to generate a summary of the given query in a human-readable format with a maximum of 500 characters, 
and to parse the given event date string to generate the start_date and end_date as Unix timestamps.
If the start_date cannot be detected, use today's date. If the end_date cannot be detected, use the end of the year date.
Return the output in JSON format as follows:
{{
    "summary": "SUMMARY",
    "start_date": UNIX,
    "end_date": UNIX
}}
Ensure the timestamps are accurate and represent the correct start and end dates of the event.
Example Input: 

    Event Date: "17 Feb - 27 Mar 2025 (Monday-Thursday)",
    Body: "Weekday Specials (Monday – Thursday)FREE Movie TicketSpend min. $80* and redeem a Golden Village movie ticket (worth $11.50)!PLUS! BONUS 1,000 Rewards+ PointsWhen you spend at least $10 at participating F&B shops on B1 as part of your $80* total spend.List of participating B1 F&B shops:Boost JuiceEdith PatisserieFun ToastGOPIZZAKei KaisendonMaki-sanNinja MamaPancake King & KopiPita Tree KebabsRollgaadiSG HawkerTea PulseThe Fish & Chips ShopToritenTyphoon CaféYum Yum ThaiSAFRA ExclusivePresent your SAFRA membership card and redeem a FREE i12 Katong umbrella with a min. spend of $80*.*Max. of 3 same-day receipts. Valid from Monday - Thursday only. Double spending required for supermarket and enrichment centres. Limited to the first 500 redemptions. Limited to 1 redemption per member per day. Redemptions must be made at the Concierge at Level 3. Other Terms & Conditions apply."

Example Output:
{{
    "summary": "Weekday Specials (Monday – Thursday)FREE Movie TicketSpend min. $80* and redeem a Golden Village movie ticket (worth $11.50)!PLUS! BONUS 1,000 Rewards+ PointsWhen you spend at least $10 at participating F&B shops on B1 as part of your $80* total spend.",
    "start_date": 1734566400,
    "end_date": 1740393600
}}
"""