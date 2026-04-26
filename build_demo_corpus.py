"""Generate a small demo corpus for the BM25 search engine.

Writes 30 short news-style articles spanning tech, business, sports,
science, climate, and culture into ./demo_corpus/. Each file's first line
is the headline; the rest is the body. Run once, then index with
`Corpus.from_directory("demo_corpus")` or query through the Streamlit app.
"""

from pathlib import Path

ARTICLES = [
    ("tech_001",
     "Electric vehicle battery breakthrough doubles range",
     "Researchers at Stanford announced a solid-state lithium battery prototype that "
     "stores nearly twice the energy of conventional lithium-ion cells while charging "
     "to eighty percent capacity in fifteen minutes. The team expects commercial "
     "electric vehicle integration within five years if manufacturing yields improve. "
     "Industry analysts called the result a meaningful step toward grid-scale "
     "energy storage as well, where battery cost remains the dominant constraint."),

    ("tech_002",
     "OpenAI releases GPT-five with stronger reasoning",
     "OpenAI shipped its next-generation language model on Tuesday, claiming "
     "substantial gains on mathematical reasoning, multi-step planning, and code "
     "generation benchmarks. Early enterprise customers reported a roughly "
     "thirty-percent reduction in hallucination rates compared to GPT-four-turbo. "
     "The model is available through the standard API at unchanged pricing for "
     "input tokens, with output tokens billed at a small premium."),

    ("tech_003",
     "Apple announces foldable iPad for 2026",
     "Apple confirmed plans to ship a foldable iPad in late 2026 at its annual "
     "developer conference. The device unfolds from a ten-inch tablet to a "
     "fourteen-inch display, runs the existing iPad operating system, and supports "
     "the Apple Pencil. Industry observers see the announcement as a hedge against "
     "slowing tablet demand and pressure from Samsung's foldable phones."),

    ("tech_004",
     "Quantum computing milestone reached at Google",
     "Google researchers reported error-corrected quantum operations on a logical "
     "qubit for the first time, a development the company called the most "
     "significant step toward fault-tolerant quantum computing in five years. "
     "The result is published in Nature; competing labs at IBM and IonQ said they "
     "expected to reproduce similar findings within months."),

    ("tech_005",
     "Cybersecurity firm warns of new ransomware strain",
     "A novel ransomware family targeting hospital electronic-health-record "
     "systems has appeared in incident reports across three states. The malware "
     "encrypts patient databases and demands payment in privacy-focused "
     "cryptocurrencies. Security researchers traced the operators to an "
     "Eastern European group previously linked to industrial-control attacks."),

    ("biz_001",
     "Federal Reserve cuts interest rates by quarter point",
     "The Federal Reserve cut its benchmark interest rate by a quarter percentage "
     "point on Wednesday, citing softening labor-market data and easing inflation. "
     "Equity markets rose on the announcement, with the S&P five hundred closing "
     "up nearly one percent. Bond yields fell as investors digested the chair's "
     "guidance that further cuts remain on the table for the next meeting."),

    ("biz_002",
     "Amazon acquires logistics startup for two billion dollars",
     "Amazon agreed to acquire same-day logistics startup Fleetly for two billion "
     "dollars in cash, the largest acquisition by the retailer since its purchase "
     "of MGM. Fleetly operates more than four hundred urban micro-fulfilment "
     "centres across North America. The deal is expected to close in the third "
     "quarter pending regulatory review."),

    ("biz_003",
     "Tesla reports record quarterly deliveries despite price cuts",
     "Tesla delivered a record number of vehicles in the first quarter despite "
     "cutting prices on its Model Y by nearly ten percent. Margins came in below "
     "Wall Street expectations as the company prioritised volume to defend market "
     "share against Chinese competitors. The company reaffirmed full-year delivery "
     "guidance and said robotaxi production remains on schedule."),

    ("biz_004",
     "Boeing fixes manufacturing defect on 737 Max",
     "Boeing said engineers had identified and corrected a torque-spec defect "
     "affecting fuselage panels on a portion of its 737 Max production line. The "
     "company will retrofit roughly sixty aircraft already in customer hands. "
     "The Federal Aviation Administration is reviewing the corrective-action plan "
     "but has not grounded the affected fleet."),

    ("biz_005",
     "Inflation cools to three-year low in March data",
     "The consumer price index rose two and one-tenth percent year over year in "
     "March, the slowest pace since early 2021. Energy prices led the deceleration; "
     "shelter costs continued to moderate but remained above the Federal Reserve's "
     "two-percent target. Economists see the print as supportive of additional "
     "rate cuts later this year."),

    ("sports_001",
     "Lakers defeat Celtics in overtime thriller",
     "The Los Angeles Lakers edged the Boston Celtics in overtime on Tuesday "
     "behind a thirty-eight-point performance from their starting forward and a "
     "buzzer-beating three-pointer in regulation. The win moves the Lakers to "
     "second in the Western Conference standings with eight games remaining."),

    ("sports_002",
     "Manchester City clinches Premier League title",
     "Manchester City secured its sixth Premier League title in seven seasons "
     "with a one-nil away win over Tottenham. The club becomes the first in "
     "English top-flight history to win the league four consecutive times. "
     "The manager praised the team's defensive discipline through a difficult "
     "spring fixture list."),

    ("sports_003",
     "Olympic swimming records broken in Paris trials",
     "Two world records fell on the opening day of Olympic swimming trials in "
     "Paris, both in the women's hundred-metre butterfly. Analysts credit the "
     "newly approved high-buoyancy swimsuit material, though the international "
     "federation maintains the suit fully complies with current regulations."),

    ("sports_004",
     "NFL announces eighteen-game regular season starting 2026",
     "The National Football League and the players' association reached a "
     "tentative agreement to expand the regular season to eighteen games "
     "starting in 2026, in exchange for an additional bye week and a higher "
     "minimum salary. The owners are scheduled to vote on the proposal next month."),

    ("science_001",
     "NASA confirms water plumes erupting from Europa",
     "NASA's Europa Clipper spacecraft confirmed active water-vapour plumes "
     "above the surface of Jupiter's moon Europa during its first close flyby. "
     "Scientists say the finding supports the case for a subsurface ocean "
     "potentially habitable to microbial life. A follow-up mission is being "
     "planned for the early 2030s."),

    ("science_002",
     "Gene therapy restores hearing in deaf children",
     "A gene therapy targeting the OTOF mutation restored functional hearing in "
     "five of six children treated in a Phase II trial, the principal investigator "
     "told a conference in Boston. The treatment is delivered via a single inner-ear "
     "injection and works only in patients with this specific genetic form of "
     "deafness, which accounts for roughly two percent of inherited hearing loss."),

    ("science_003",
     "Climate scientists report record Arctic sea ice loss",
     "Arctic sea ice extent reached its lowest March measurement on record this "
     "year, continuing a long-term decline of more than thirteen percent per "
     "decade. The thinning poses risks for shipping routes, polar wildlife, and "
     "global weather patterns. The lead author called the trajectory consistent "
     "with high-emission climate scenarios."),

    ("science_004",
     "Fusion startup hits net-energy-gain milestone",
     "Commonwealth Fusion Systems reported its tokamak achieved a brief net-energy "
     "gain in a controlled fusion shot, the first private-sector demonstration of "
     "the milestone. The company said commercial reactors remain at least a decade "
     "away, but the result strengthens its case for a planned utility-scale plant "
     "in Virginia."),

    ("science_005",
     "AI model predicts protein interactions across the proteome",
     "A new deep-learning model from DeepMind predicts how every human protein "
     "interacts with every other, producing the most comprehensive interaction "
     "map to date. Researchers expect the resource to accelerate drug discovery, "
     "particularly for conditions where the relevant biological pathway has "
     "been hard to characterise experimentally."),

    ("climate_001",
     "European countries agree on carbon-border adjustment expansion",
     "The European Union expanded its carbon-border adjustment mechanism to "
     "cover steel, aluminium, fertiliser, and cement imports starting next year. "
     "Trading partners criticised the measure as a disguised tariff; EU "
     "officials maintain it merely levels carbon costs between domestic "
     "manufacturers and importers."),

    ("climate_002",
     "Solar overtakes coal in US electricity mix",
     "Solar generation surpassed coal-fired generation in the United States for "
     "the first time on a quarterly basis, the Energy Information Administration "
     "reported. The shift reflects continued solar capacity additions and "
     "accelerated coal retirements driven by economic and regulatory pressure. "
     "Wind generation remained the largest single renewable contributor."),

    ("politics_001",
     "Senate passes infrastructure modernisation bill",
     "The Senate passed a six-hundred-billion-dollar infrastructure bill on a "
     "bipartisan vote of seventy-two to twenty-six. The package funds road and "
     "bridge repairs, broadband expansion in rural areas, and electric-vehicle "
     "charging deployment. The bill now goes to the House for consideration."),

    ("politics_002",
     "Supreme Court hears arguments on platform liability",
     "The Supreme Court heard oral arguments in a case that could reshape "
     "the legal protections enjoyed by social-media platforms under Section "
     "two-thirty. Justices appeared sceptical of arguments that recommendation "
     "algorithms should be treated identically to neutral hosting. A decision "
     "is expected by June."),

    ("politics_003",
     "Election officials warn of deepfake interference",
     "State election officials issued a joint advisory warning that AI-generated "
     "deepfakes targeting voters and poll workers had increased markedly during "
     "the spring primary season. Officials urged voters to verify time-sensitive "
     "information through official channels rather than social media."),

    ("culture_001",
     "Streaming service announces price increase for ad-free tier",
     "A major streaming service raised the monthly price of its ad-free tier by "
     "two dollars, citing rising content costs. The cheaper ad-supported tier "
     "saw no change. The company said churn was modest in test markets that "
     "received the price increase last quarter."),

    ("culture_002",
     "Director of acclaimed film wins international award",
     "The Cannes Film Festival jury awarded its Palme d'Or to a quiet character "
     "study about an elderly piano tuner. Critics described the film as the most "
     "patient and humane work in the festival lineup. The director thanked the "
     "small Italian town that hosted the production."),

    ("health_001",
     "FDA approves once-weekly insulin for type-two diabetes",
     "The Food and Drug Administration approved a once-weekly basal insulin "
     "injection for adults with type-two diabetes, ending decades of daily-only "
     "insulin therapy. Trials showed comparable glucose control with fewer "
     "self-reported lapses in adherence. The drug will be available next quarter "
     "at a list price comparable to existing daily formulations."),

    ("health_002",
     "Telehealth visits remain elevated post-pandemic",
     "Telehealth visits accounted for roughly fifteen percent of outpatient "
     "encounters last year, well above the pre-pandemic baseline of less than "
     "one percent. Mental-health and follow-up consultations dominate virtual "
     "use; in-person care remains preferred for new diagnostic encounters."),

    ("retail_001",
     "Holiday retail sales beat expectations on apparel rebound",
     "Holiday retail sales rose four and a half percent year over year, "
     "exceeding industry forecasts. Apparel led the surprise on the strength "
     "of cold winter weather across the northern states. Online sales grew "
     "nine percent and now account for nearly thirty percent of holiday spending."),

    ("retail_002",
     "Grocery chains pilot AI-driven dynamic pricing",
     "Two large grocery chains began pilots of AI-driven dynamic pricing on "
     "fresh produce and bakery items, adjusting shelf prices in near-real-time "
     "based on inventory and projected demand. Consumer-advocacy groups warned "
     "that opaque pricing could disadvantage shoppers; the chains say the "
     "system reduces food waste."),
]


def main(out_dir: str = "demo_corpus") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    for stem, headline, body in ARTICLES:
        (out / f"{stem}.txt").write_text(f"{headline}\n{body}\n", encoding="utf-8")
    print(f"Wrote {len(ARTICLES)} articles to {out}/")


if __name__ == "__main__":
    main()
