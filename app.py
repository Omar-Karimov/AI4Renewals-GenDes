
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from AI4Renewals.constant import APP_HOST, APP_PORT
from AI4Renewals.pipeline.prediction_pipeline import GenDesData, GenDesClassifier
from AI4Renewals.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Country: Optional[str] = None
        self.user_education: Optional[str] = None
        self.Gender: Optional[str] = None
        self.LastLoginDaysAgo: Optional[str] = None
        self.Profession: Optional[str] = None
        self.SubscriptionType: Optional[str] = None
        self.UsesVR: Optional[str] = None
        self.Age: Optional[str] = None
        self.DesignProjectScale: Optional[str] = None
        self.NumberOfDesigns: Optional[str] = None
        self.TutorialProgressionType: Optional[str] = None
        self.FrequencyOfDesignToolUsage: Optional[str] = None
        self.CustomerSupportCall: Optional[str] = None
        self.AppEngagementMinutes: Optional[str] = None
        self.IsActive: Optional[str] = None
        

    async def get_gendes_data(self):
        form = await self.request.form()
        self.Country = form.get("Country")
        self.user_education = form.get("user_education")
        self.Gender = form.get("Gender")
        self.LastLoginDaysAgo = form.get("LastLoginDaysAgo")
        self.Profession = form.get("Profession")
        self.SubscriptionType = form.get("SubscriptionType")
        self.UsesVR = form.get("UsesVR")
        self.Age = form.get("Age")
        self.DesignProjectScale = form.get("DesignProjectScale")
        self.NumberOfDesigns = form.get("NumberOfDesigns")
        self.TutorialProgressionType = form.get("TutorialProgressionType")
        self.FrequencyOfDesignToolUsage = form.get("FrequencyOfDesignToolUsage")
        self.CustomerSupportCall = form.get("CustomerSupportCall")
        self.AppEngagementMinutes = form.get("AppEngagementMinutes")
        self.IsActive = form.get("IsActive")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "gendes.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_gendes_data()
        
        gendes_data = GenDesData(
                                Country = form.Country,
                                user_education = form.user_education,
                                Gender = form.Gender,
                                LastLoginDaysAgo = form.LastLoginDaysAgo,
                                Profession= form.Profession,
                                SubscriptionType = form.SubscriptionType,
                                UsesVR = form.UsesVR,
                                Age = form.Age,
                                DesignProjectScale = form.DesignProjectScale,
                                NumberOfDesigns = form.NumberOfDesigns,
                                TutorialProgressionType = form.TutorialProgressionType,
                                FrequencyOfDesignToolUsage = form.FrequencyOfDesignToolUsage,
                                CustomerSupportCall = form.CustomerSupportCall,
                                AppEngagementMinutes = form.AppEngagementMinutes,
                                IsActive = form.IsActive,
                                )
        
        gendes_df = gendes_data.get_gendes_input_data_frame()

        model_predictor = GenDesClassifier()

        value = model_predictor.predict(dataframe=gendes_df)[0]

        status = None
        if value == 1:
            status = "User will not renew subscription"
        else:
            status = "User will renew subscription"

        return templates.TemplateResponse(
            "gendes.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)