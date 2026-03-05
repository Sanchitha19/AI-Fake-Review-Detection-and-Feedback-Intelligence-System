import pandas as pd
import random

# Configuration
NUM_ROWS = 300
FILENAME = "reviews.csv"

# Patterns
fake_templates = [
    "This is the {adj} product ever! {call_to_action} for an {deal}.",
    "{call_to_action}: {adj} product ever. {deal}!",
    "I've never seen such an {deal}. {call_to_action} now, {adj} item.",
    "Unbelievable! {adj} product ever. {call_to_action} quickly.",
    "Don't miss out! {deal} on the {adj} product ever. {call_to_action}."
]

adjectives = ["best", "greatest", "most amazing", "incredible", "top-rated"]
calls_to_action = ["Buy now", "Get it now", "Shop today", "Order immediately", "Limited offer"]
deals = ["amazing deal", "huge discount", "limited time offer", "special price", "unbeatable bargain"]

genuine_templates = [
    "I am having {issue} with this device. {result}.",
    "The {part} is {quality}, but {negative_aspect}.",
    "I was disappointed by the {issue}. {result}.",
    "Overall, the {part} works well. Unfortunately, {issue}.",
    "Great {part}, however I experienced {issue} recently."
]

issues = ["battery issues", "delivery delays", "software bugs", "overheating problems", "connectivity drops"]
results = ["It's very frustrating", "I might return it", "Hope it gets fixed", "Not what I expected", "Could be better"]
parts = ["screen", "camera", "battery", "build quality", "user interface"]
qualities = ["excellent", "decent", "crisp", "solid", "premium"]
negative_aspects = ["the delivery was late", "the price is too high", "it feels a bit heavy", "customer service was slow", "it lack some features"]

data = []

for _ in range(NUM_ROWS // 2):
    # Fake review
    template = random.choice(fake_templates)
    review = template.format(
        adj=random.choice(adjectives),
        call_to_action=random.choice(calls_to_action),
        deal=random.choice(deals)
    )
    data.append({"review_text": review, "label": "fake"})
    
    # Genuine review
    template = random.choice(genuine_templates)
    review = template.format(
        issue=random.choice(issues),
        result=random.choice(results),
        part=random.choice(parts),
        quality=random.choice(qualities),
        negative_aspect=random.choice(negative_aspects)
    )
    data.append({"review_text": review, "label": "genuine"})

# Shuffle and save
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(FILENAME, index=False)

print(f"Dataset generated with {len(df)} rows and saved to {FILENAME}")
