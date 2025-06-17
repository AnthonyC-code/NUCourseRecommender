# file: handler.py   (~30 lines)
import json, os, boto3
DDB   = boto3.resource("dynamodb")
RECS  = DDB.Table(os.environ["RECOMMEND_TABLE"])
META  = DDB.Table(os.environ["META_TABLE"])

def lambda_handler(event, _ctx):
    q = event.get("queryStringParameters") or {}
    cid = q.get("course")
    k   = int(q.get("k", "5"))
    if not cid:
        return {"statusCode": 400,
                "body": "missing ?course="}

    entry = RECS.get_item(Key={"course_id": cid}).get("Item")
    if not entry:
        return {"statusCode": 404, "body": "course not found"}

    out = []
    for c in entry["neighbors"][:k]:
        meta = META.get_item(Key={"course_id": c}).get("Item")
        if meta:
            meta.pop("course_id", None)
            out.append(meta)
    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(out)}
