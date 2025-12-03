from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import json
import re

import requests
from bs4 import BeautifulSoup

# Optional NLP tools for methods/tools identification.
# If NLTK is not installed, the parser will still work using simple
# keyword lists; the WordNet-based expansion is just a bonus.
try:  # pragma: no cover - optional dependency
    import nltk  # type: ignore
    from nltk.corpus import wordnet as wn  # type: ignore
    from nltk.stem import WordNetLemmatizer  # type: ignore
    from nltk import pos_tag as _nltk_pos_tag, word_tokenize as _nltk_word_tokenize  # type: ignore
except Exception:  # pragma: no cover
    wn = None
    WordNetLemmatizer = None
    _nltk_pos_tag = None
    _nltk_word_tokenize = None


@dataclass
class Ingredient:
    raw: str
    name: str
    quantity: Optional[float]
    unit: Optional[str]
    descriptor: Optional[str]
    preparation: Optional[str]


@dataclass
class Step:
    step_number: int
    description: str   
    ingredients: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    time: Dict[str, str] = field(default_factory=dict)
    temperature: Dict[str, str] = field(default_factory=dict)
    action: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    modifiers: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, str] = field(default_factory=dict)


@dataclass
class Recipe:
    title: str
    url: str
    ingredients: List[Ingredient]
    tools: List[str]
    methods: List[str]
    steps: List[Step]




UNITS = [
    "#",
    "#s",
    "bag",
    "bags",
    "bottle",
    "bottles",
    "bunch",
    "bunches",
    "c",
    "can",
    "cans",
    "clove",
    "cloves",
    "cs",
    "cube",
    "cubes",
    "cup",
    "cups",
    "dash",
    "dashes",
    "dessertspoon",
    "dessertspoons",
    "envelope",
    "envelopes",
    "fl oz",
    "fl ozs",
    "fluid ounce",
    "fluid ounces",
    "fluid oz",
    "fluid ozs",
    "gal",
    "gallon",
    "gallons",
    "gals",
    "gram",
    "grams",
    "head",
    "heads",
    "inch",
    "inches",
    "jar",
    "jars",
    "kilogram",
    "kilograms",
    "lb",
    "lbs",
    "liter",
    "liters",
    "milliliter",
    "milliliters",
    "ml",
    "mls",
    "ounce",
    "ounces",
    "oz",
    "ozs",
    "package",
    "packages",
    "packet",
    "packets",
    "piece",
    "pieces",
    "pinch",
    "pinches",
    "pint",
    "pints",
    "pound",
    "pounds",
    "pt",
    "pts",
    "qt",
    "qts",
    "quart",
    "quarts",
    "sheet",
    "sheets",
    "slice",
    "slices",
    "strip",
    "strips",
    "tablespoon",
    "tablespoons",
    "Tbsp",
    "Tbsps",
    "teaspoon",
    "teaspoons",
    "tsp",
    "tsps",
]

DESCRIPTORS = [
    "fresh",
    "freshly",
    "frozen",
    "chilled",
    "cold",
    "cool",
    "warm",
    "lukewarm",
    "hot",
    "room temperature",
    "refrigerated",
    "thawed",
    "dried",
    "dry",
    "dry roasted",
    "tender",
    "tough",
    "soft",
    "firm",
    "chewy",
    "crunchy",
    "crispy",
    "crumbly",
    "crusty",
    "fluffy",
    "dense",
    "smooth",
    "creamy",
    "syrupy",
    "bubbly",
    "sweet",
    "sweetened",
    "unsweetened",
    "bitter",
    "salty",
    "savory",
    "acidic",
    "tangy",
    "sour",
    "spicy",
    "mild",
    "fiery",
    "earthy",
    "smoky",
    "rich",
    "buttery",
    "lean",
    "meaty",
    "fatty",
    "nonfat",
    "low sodium",
    "reduced sodium",
    "unsalted",
    "extra virgin",
    "rare",
    "medium rare",
    "medium-rare",
    "medium",
    "well done",
    "gamey",
    "juicy",
    "whole",
    "halved",
    "quartered",
    "cubed",
    "diced",
    "minced",
    "chopped",
    "finely chopped",
    "coarsely chopped",
    "sliced",
    "thinly sliced",
    "thick-cut",
    "julienned",
    "matchstick cut",
    "shredded",
    "crumbled",
    "crushed",
    "ground",
    "mashed",
    "peeled",
    "pitted",
    "seeded",
    "cored",
    "torn",
    "roughly chopped",
    "baked",
    "boiled",
    "blanched",
    "braised",
    "broiled",
    "browned",
    "charred",
    "caramelized",
    "fried",
    "pan-fried",
    "deep-fried",
    "grilled",
    "roasted",
    "toasted",
    "steamed",
    "simmered",
    "stewed",
    "sauteed",
    "stir-fried",
    "canned",
    "candied",
    "smoked",
    "cured",
    "pickled",
    "fermented",
    "freeze dried",
    "dried out",
    "boneless",
    "bony",
    "skinless",
    "meaty",
    "moist",
    "juicy",
    "superfine",
    "organic",
    "natural",
    "fragrant",
    "aromatic",
]

TOOLS = [
    "knife",
    "chef knife",
    "chef's knife",
    "chefs knife",
    "paring knife",
    "bread knife",
    "utility knife",
    "carving knife",
    "steak knife",
    "santoku knife",
    "cutting board",
    "bench scraper",
    "dough scraper",
    "peeler",
    "vegetable peeler",
    "grater",
    "cheese grater",
    "nutmeg grater",
    "microplane",
    "zester",
    "nutcracker",
    "spoon",
    "wooden spoon",
    "slotted spoon",
    "spatula",
    "fish spatula",
    "burger spatula",
    "turner",
    "tongs",
    "silicone tong",
    "whisk",
    "balloon whisk",
    "flat whisk",
    "french whisk",
    "mixing whisk",
    "bowl",
    "mixing bowl",
    "cup",
    "measuring cup",
    "measuring jar",
    "measuring jug",
    "measuring spoon",
    "food storage container",
    "frying pan",
    "skillet",
    "grill pan",
    "griddle",
    "saucepan",
    "pot",
    "stockpot",
    "clay pot",
    "beanpot",
    "mated colander pot",
    "mortar",
    "molcajete",
    "baking sheet",
    "baking dish",
    "cake pan",
    "loaf pan",
    "muffin tin",
    "pie dish",
    "pie server",
    "pie cutter",
    "pizza cutter",
    "pizza shovel",
    "pizza slicer",
    "rolling pin",
    "pastry bag",
    "pastry brush",
    "pastry blender",
    "pastry wheel",
    "cookie cutter",
    "cookie mould",
    "cookie press",
    "biscuit cutter",
    "biscuit mould",
    "biscuit press",
    "colander",
    "sieve",
    "drum sieve",
    "strainer",
    "spider",
    "spider strainer",
    "spoon skimmer",
    "spoon sieve",
    "blender",
    "food mill",
    "food processor",
    "coffee grinder",
    "burr grinder",
    "burr mill",
    "milk frother",
    "garlic press",
    "citrus reamer",
    "lemon reamer",
    "lemon squeezer",
    "cherry pitter",
    "apple corer",
    "apple cutter",
    "mandoline",
    "ice cream scoop",
    "melon baller",
    "egg slicer",
    "egg separator",
    "thermometer",
    "meat thermometer",
    "candy thermometer",
    "kitchen scale",
    "weighing scales",
    "timer",
    "oven",
    "stove",
    "oven mitt",
    "oven glove",
    "pot holder",
    "potholder",
    "trussing needle",
    "kitchen twine",
    "cooking twine",
    "butcher's twine",
]

PRIMARY_METHODS = [
    "bake",
    "boil",
    "broil",
    "fry",
    "deep-fry",
    "pan-fry",
    "stir-fry",
    "braise",
    "roast",
    "grill",
    "steam",
    "stew",
    "simmer",
    "saute",
    "searing",
    "pressure cook",
    "blend",
    "mix",
    "poach",
]

OTHER_METHODS = [
    "beat",
    "brown",
    "brush",
    "chill",
    "combine",
    "cool",
    "cover",
    "cream",
    "crumble",
    "cut",
    "dip",
    "drain",
    "flip",
    "flour",
    "fold",
    "garnish",
    "grease",
    "heat",
    "layer",
    "line",
    "mash",
    "measure",
    "melt",
    "mix",
    "pound",
    "pour",
    "preheat",
    "refrigerate",
    "rinse",
    "season",
    "serve",
    "shake",
    "sift",
    "simmer",
    "slice",
    "soak",
    "spoon",
    "spread",
    "sprinkle",
    "stir",
    "strain",
    "stuff",
    "toast",
    "toss",
    "turn",
    "whisk",
    "chop",
    "mince",
    "dice",
    "julienne",
    "shred",
    "grate",
    "knead",
    "marinate",
]










def _init_lemmatizer():
    if WordNetLemmatizer is None:
        return None
    try:
        return WordNetLemmatizer()
    except Exception:
        return None


_LEMMATIZER = _init_lemmatizer()

def _lemmatize(token: str, pos: str) -> str:

    t = token.lower()
    if _LEMMATIZER is not None and pos in {"n", "v"}:
        try:
            return _LEMMATIZER.lemmatize(t, pos)
        except Exception:
            pass


    if pos == "v":
        for suf in ["ing", "ed", "es", "s"]:
            if t.endswith(suf) and len(t) > len(suf) + 1:
                return t[: -len(suf)]
    
    if pos == "n" and t.endswith("s") and len(t) > 3:
        return t[:-1]
    return t


def _build_cooking_verbs_from_wordnet() -> List[str]:
    if wn is None:
        return []
    seeds = ["cook", "bake", "boil", "fry", "roast", "grill", "steam", "saute", "sauté"]
    verbs = set()
    
    for seed in seeds:
        try:
            for syn in wn.synsets(seed, pos="v"):
                for lemma in syn.lemmas():
                    verbs.add(lemma.name().replace("_", " ").lower())
        except Exception:
            break
    return sorted(verbs)


COOKING_VERBS = set(PRIMARY_METHODS + OTHER_METHODS)
COOKING_VERBS.update(_build_cooking_verbs_from_wordnet())


TOOL_LEMMA_TO_CANONICAL: Dict[str, str] = {}
for _tool in TOOLS:
    lemma = _lemmatize(_tool, "n")
    TOOL_LEMMA_TO_CANONICAL.setdefault(lemma, _tool)




def fetch_html(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def parse_allrecipes_basic(html: str) -> Dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    json_ld = soup.find("script", type="application/ld+json")
    if json_ld is None or not json_ld.string:
        raise ValueError("Could not find recipe JSON-LD on page.")

    data = json.loads(json_ld.string)
    if isinstance(data, list):
        recipe_obj = None
        for item in data:
            t = item.get("@type")
            if t == "Recipe" or (isinstance(t, list) and "Recipe" in t):
                recipe_obj = item
                break

        if recipe_obj is None:
            raise ValueError("JSON-LD does not contain a Recipe object.")
        data = recipe_obj

    title = data.get("name", "Unknown recipe")
    raw_ingredients = data.get("recipeIngredient", [])
    instructions = data.get("recipeInstructions", [])

    steps_raw: List[str] = []
    for inst in instructions:
        if isinstance(inst, dict):
            text = inst.get("text", "")
            if text:
                steps_raw.append(text.strip())
        elif isinstance(inst, str):
            if inst.strip():
                steps_raw.append(inst.strip())

    return {
        "title": title,
        "ingredients_raw": raw_ingredients,
        "steps_raw": steps_raw,
    }



def parse_quantity(token: str) -> Optional[float]:
    """
    Parse a quantity token into a float, handling fractions like "1/2"
    and mixed forms like "1-1/2".
    """
    token = token.strip()
    # simple float
    try:
        return float(token)
    except ValueError:
        pass


    if "/" in token and "-" not in token:
        parts = token.split("/")
        if len(parts) == 2:
            num, den = parts
            try:
                return float(num) / float(den)
            except ValueError:
                return None

    if "-" in token:
        parts = token.split("-")
        if len(parts) == 2:
            base_str, frac_str = parts
            try:
                base = float(base_str)
            except ValueError:
                base = 0.0
            frac = parse_quantity(frac_str)
            if frac is not None:
                return base + frac

    return None



def parse_ingredient_line(line: str) -> Ingredient:
    raw = line.strip()
    tokens = raw.split()
    quantity: Optional[float] = None
    unit: Optional[str] = None
    descriptor_tokens: List[str] = []
    name_tokens: List[str] = []
    preparation: Optional[str] = None

    # 1. Quantity
    if tokens:
        q = parse_quantity(tokens[0])
        if q is not None:
            quantity = q
            tokens = tokens[1:]

    # 2. Unit
    if tokens and tokens[0].lower() in UNITS:
        unit = tokens[0].lower()
        tokens = tokens[1:]

    # 3. Split into "before comma" and "preparation after comma"
    before_comma, *after_comma = " ".join(tokens).split(",", 1)
    if after_comma:
        prep_str = after_comma[0].strip()
        if prep_str:
            preparation = prep_str

    # 4. Separate descriptors from name
    for tok in before_comma.split():
        if tok.lower() in DESCRIPTORS:
            descriptor_tokens.append(tok.lower())
        else:
            name_tokens.append(tok)

    descriptor = " ".join(descriptor_tokens) if descriptor_tokens else None
    name = " ".join(name_tokens).strip()

    return Ingredient(
        raw=raw,
        name=name,
        quantity=quantity,
        unit=unit,
        descriptor=descriptor,
        preparation=preparation,
    )


def parse_ingredients(raw_ingredients: List[str]) -> List[Ingredient]:
    return [parse_ingredient_line(line) for line in raw_ingredients]




def split_into_atomic_steps(step_text: str) -> List[str]:
    parts = re.split(r"[.;]", step_text)
    return [p.strip() for p in parts if p.strip()]


def find_items_in_text(text: str, vocab: List[str]) -> List[str]:
    text_lower = text.lower()
    found: List[str] = []
    for word in vocab:
        if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
            found.append(word)
    return found


def extract_time(text: str) -> Dict[str, str]:
    match = re.search(r"(\d+)\s*(minutes?|mins?|hours?|hrs?)", text, flags=re.I)
    if match:
        return {"duration": match.group(0)}
    return {}


def extract_temperature(text: str) -> Dict[str, str]:
    match = re.search(r"(\d+)\s*(degrees\s*)?(F|C)", text, flags=re.I)
    if match:
        return {"oven": match.group(0)}
    return {}


def _pos_tag(text: str):
    if _nltk_word_tokenize is not None and _nltk_pos_tag is not None:
        try:
            tokens = _nltk_word_tokenize(text)
            return _nltk_pos_tag(tokens)
        except Exception:
            pass
    # Fallback: simple word split with fake verb tags
    tokens = re.findall(r"[A-Za-z]+", text)
    return [(t, "VB") for t in tokens]


def extract_cooking_methods(text: str) -> List[str]:
    tagged = _pos_tag(text)
    methods: List[str] = []

    for token, tag in tagged:
        if tag.startswith("VB"):
            lemma = _lemmatize(token, "v")
            if lemma in COOKING_VERBS:
                methods.append(lemma)


    text_lower = text.lower()
    for verb in PRIMARY_METHODS + OTHER_METHODS:
        if " " in verb and verb in text_lower:
            methods.append(verb)

    seen = set()
    unique: List[str] = []
    
    for m in methods:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def extract_tools_from_text(text: str) -> List[str]:
    tagged = _pos_tag(text)
    tools: List[str] = []

    for token, tag in tagged:
        if tag.startswith("NN"):
            lemma = _lemmatize(token, "n")
            if lemma in TOOL_LEMMA_TO_CANONICAL:
                tools.append(TOOL_LEMMA_TO_CANONICAL[lemma])

    tools.extend(find_items_in_text(text, TOOLS))
    seen = set()
    unique: List[str] = []
    
    for t in tools:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique



def build_steps(steps_raw: List[str], ingredients: List[Ingredient]) -> List[Step]:
    steps: List[Step] = []
    ingredient_names = [ing.name.lower() for ing in ingredients]

    current_oven_temp: Optional[str] = None
    step_counter = 1

    for raw_step in steps_raw:
        atomic_texts = split_into_atomic_steps(raw_step)
        for text in atomic_texts:
            tools = extract_tools_from_text(text)
            methods = extract_cooking_methods(text)

            time_info = extract_time(text)
            temp_info = extract_temperature(text)


            if "oven" in temp_info:
                current_oven_temp = temp_info["oven"]

            used_ingredients: List[str] = []
            text_lower = text.lower()

            for name in ingredient_names:
                if name and re.search(r"\b" + re.escape(name) + r"\b", text_lower):
                    used_ingredients.append(name)

            action = methods[0] if methods else None

            context: Dict[str, str] = {}
            if current_oven_temp:
                context["oven_temperature"] = current_oven_temp

            step = Step(
                step_number=step_counter,
                description=text,
                ingredients=used_ingredients,
                tools=tools,
                methods=methods,
                time=time_info,
                temperature=temp_info,
                action=action,
                objects=used_ingredients,
                modifiers={"tools": ", ".join(tools)} if tools else {},
                context=context,
            )
            steps.append(step)
            step_counter += 1

    return steps



def collect_recipe_tools_and_methods(steps: List[Step]) -> Tuple[List[str], List[str]]:
    tools_set = set()
    methods_set = set()
    for s in steps:
        tools_set.update(s.tools)
        methods_set.update(s.methods)
    return sorted(tools_set), sorted(methods_set)

def parse_recipe_from_url(url: str) -> Recipe:
    html = fetch_html(url)
    base = parse_allrecipes_basic(html)

    ingredients = parse_ingredients(base["ingredients_raw"])
    steps = build_steps(base["steps_raw"], ingredients)
    tools, methods = collect_recipe_tools_and_methods(steps)

    return Recipe(
        title=base["title"],
        url=url,
        ingredients=ingredients,
        tools=tools,
        methods=methods,
        steps=steps,
    )


if __name__ == "__main__":
    test_url = input("Enter an AllRecipes URL: ").strip()
    recipe = parse_recipe_from_url(test_url)
    print("Title:", recipe.title)
    print("Number of ingredients:", len(recipe.ingredients))
    print("Number of steps:", len(recipe.steps))
