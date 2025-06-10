from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["main"])
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Serve the test page"""
    return templates.TemplateResponse("test.html", {"request": request})

@router.get("/about", response_class=HTMLResponse)
async def get_about_page(request: Request):
    """Serve the about page"""
    return templates.TemplateResponse("about.html", {"request": request}) 