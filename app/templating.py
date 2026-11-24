from pathlib import Path

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def euro(value) -> str:
    """1234.5 -> '1.234,50 €'"""
    if value is None:
        return "–"
    s = f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} €"


def datum(value) -> str:
    """'2024-11-03' -> '03.11.2024'"""
    if not value:
        return "–"
    try:
        y, m, d = str(value)[:10].split("-")
        return f"{d}.{m}.{y}"
    except ValueError:
        return str(value)


templates.env.filters["euro"] = euro
templates.env.filters["datum"] = datum
