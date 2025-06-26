import io
import json
import os

from dotenv import load_dotenv

load_dotenv()

bucket = os.getenv("BUCKET_URL")
region = os.getenv("REGION")
object_key = os.getenv("OBJECT_KEY")


#values = ["1617", "2512", "20.7", "Necesita riego"]
#values = ["1444","785","23.0","Marchita"]
values = ["2007","1084","21.0","Saludable"]

buffer = io.StringIO()
data_dict = {
    "soil": values[0],
    "light": values[1],
    "temp_c": values[2],
    "result": values[3]
}
data = io.StringIO()
json.dump(data_dict, data)
data = data.getvalue().encode('utf-8')

# Upload to S3
url = f'https://{bucket}.s3.{region}.amazonaws.com/{object_key}'
headers = {'Content-Type': 'application/json'}

with httpx.Client() as client:
    response = client.put(url, content=data, headers=headers)
    print(response.status_code, response.text)
