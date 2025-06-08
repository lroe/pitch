from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Status": "OK", "Message": "Root is working"}

@app.get("/login/google")
def test_login_route():
    return {"Status": "OK", "Message": "Login route is working"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
