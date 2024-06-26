from google_auth_oauthlib.flow import InstalledAppFlow

# Define the scopes required for Google Drive API access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Function to perform OAuth 2.0 authorization flow and obtain refresh token
def get_refresh_token(client_id, client_secret):
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials.json', scopes=SCOPES)
    flow.run_local_server(port=0)
    return flow.credentials.refresh_token

client_id = '****'
client_secret = '***'

# Obtain refresh token
refresh_token = get_refresh_token(client_id, client_secret)

print("Refresh token:", refresh_token)
