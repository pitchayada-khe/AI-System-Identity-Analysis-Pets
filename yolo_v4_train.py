from inference_sdk import InferenceHTTPClient
from PIL import Image
import requests
from io import BytesIO

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="WtmiXSIstHSCGCUOX6GJ"
)

image_name = "cat_test.jpg"

result = client.run_workflow(
    workspace_name="cat-dog-ugcs1",
    workflow_id="detect-count-and-visualize-4",
    images={
        "image": f"images/{image_name}"
    },
    use_cache=True
)
# print(result)
# print(f"Name : {image_name}")

if isinstance(result, list) and len(result) > 0:
    inner = result[0].get('predictions', {})
    pred = inner.get('predictions', []) 

    if not pred:
        print("No objects were detected in the image!")
    else:
        for data in pred:
            print(f"Detected : {data['class']} {data['confidence']:.2f} at ({data['x']}, {data['y']}, {data['width']}, {data['height']})")
            
            # output_image_url = result[0].get('output_image')
    
            # if output_image_url:
            #     print("Visualization image URL:", output_image_url)
            #     response = requests.get(output_image_url)
            #     img = Image.open(BytesIO(response.content))
            #     img.show()
            # else:
            #     print("No visualization image found!")

else:
    print("No predictions found!")
