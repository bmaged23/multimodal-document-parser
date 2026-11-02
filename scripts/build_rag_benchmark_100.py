"""Build source-bound questions and freeze group-separated 30/70 development/test splits."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from multimodal_doc_parser.evaluation.rag_benchmark import (
    BenchmarkQuestion,
    EvidenceNeed,
    digest,
    save,
    validate_dataset,
)


class Sources:
    def __init__(self):
        self.documents = {}
        self.children = {}
        self.parents = {}
        for name in ["original", "irena", "imf"]:
            self.documents[name] = json.loads(
                Path(
                    f"outputs/parser_runs/{name}-final-source-checked-03/document.json"
                ).read_text()
            )
            root = Path(f"outputs/rag_bundles/{name}-adaptive-parent-child-02")
            self.children[name] = [
                json.loads(line) for line in (root / "children.jsonl").read_text().splitlines()
            ]
            self.parents[name] = {
                p["id"]: p
                for p in [
                    json.loads(line) for line in (root / "parents.jsonl").read_text().splitlines()
                ]
            }

    def element(self, name, eid):
        return next(e for e in self.documents[name]["elements"] if e["id"] == eid)

    def need(self, name, eid, cells=None, series=None, labels=None, text=None):
        e = self.element(name, eid)
        accepted = []
        for child in self.children[name]:
            parent = self.parents[name][child["parent_id"]]
            parts = {p["id"]: p for p in parent["parts"]}
            selected = [parts[p["part_id"]] for p in child["slices"]] + [
                parts[p] for p in child["context_part_ids"]
            ]
            relevant = [p for p in selected if eid in p["source_element_ids"]]
            if not relevant:
                continue
            covered = {
                (c["row"], c["column"])
                for p in relevant
                for c in p["cells"]
                if c["element_id"] == eid
            }
            if cells:
                # Row search text repeats the full column-header path even
                # when the separate header record is deferred to the parent.
                for cell in e.get("cells", []):
                    if (
                        cell.get("is_header")
                        and cell["text"]
                        and any(
                            p["role"] == "table_row" and cell["text"] in p["text"] for p in relevant
                        )
                    ):
                        covered.add((cell["row"], cell["column"]))
                if not set(cells).issubset(covered):
                    continue
            if series:
                covered_labels = set()
                for p in relevant:
                    if p["role"] != "chart_series":
                        continue
                    v = json.loads(p["text"])
                    if v.get("series") == series:
                        covered_labels.add(v["label"])
                    if v.get("name") == series:
                        covered_labels.update(x["label"] for x in v["points"])
                # Different children may hold the two endpoints. One evidence
                # group is created per printed point below, never an impossible
                # requirement that both endpoints be in one child.
                if not set(labels or []).issubset(covered_labels):
                    continue
            if e["type"] in {"text", "image"} and not any(
                eid in parts[s["part_id"]]["source_element_ids"]
                and not parts[s["part_id"]]["context_only"]
                for s in child["slices"]
            ):
                continue
            accepted.append(child)
        if not accepted:
            raise ValueError(f"No expected child covers {name}/{eid}/{cells}/{series}/{labels}")
        if text is None:
            if cells:
                lookup = {(c["row"], c["column"]): c["text"] for c in e["cells"]}
                text = (
                    (e.get("caption") or "Source table")
                    + "\nHeaders: "
                    + "; ".join(
                        f"cell {c['row']},{c['column']}: {c['text']}"
                        for c in e["cells"]
                        if c.get("is_header")
                    )
                    + "\nSelected cells: "
                    + "; ".join(f"cell {r},{c}: {lookup[(r, c)]}" for r, c in cells)
                )
            elif series:
                points = next(s for s in e["series"] if s["name"] == series)
                text = json.dumps(
                    dict(
                        series=series,
                        unit=points["unit"],
                        points=[p for p in points["points"] if p["label"] in labels],
                    )
                )
            else:
                text = e.get("text") or e.get("description", {}).get("text", "")
        return EvidenceNeed(
            document_id=self.documents[name]["document_id"],
            element_id=eid,
            pages=sorted({r["page_number"] for r in e["source_regions"]}),
            cells=cells or [],
            series=series,
            point_labels=labels or [],
            accepted_child_ids=[c["id"] for c in accepted],
            accepted_parent_ids=list(dict.fromkeys(c["parent_id"] for c in accepted)),
            source_text=text,
        )


def make_questions(source):
    result = []

    def add(
        category,
        question,
        facts,
        evidence,
        group,
        numbers=None,
        status="source-bound silver reference; no independent human adjudication",
    ):
        result.append(
            dict(
                id=f"q{len(result) + 1:03d}",
                category=category,
                question=question,
                reference=" ".join(facts),
                required_facts=facts,
                expected_abstention=category == "unanswerable",
                evidence=evidence,
                group_id=group,
                split="test",
                reference_status=status,
                numeric_values=numbers or [],
            )
        )

    # Questions are authored from source passages, not from the retriever's hits.
    narrative = [
        (
            "irena",
            "element_00012",
            "Which geopolitical risks could raise renewable technology costs in the short term?",
            [
                (
                    "Trade tariffs on renewable components and materials, and Chinese "
                    "manufacturing sector dynamics."
                )
            ],
            "irena-risk",
        ),
        (
            "irena",
            "element_00013",
            "Which factors shape capital costs and renewable project viability?",
            [
                (
                    "Revenue certainty, capital structure and macroeconomic conditions "
                    "shape capital costs."
                )
            ],
            "irena-financing",
        ),
        (
            "irena",
            "element_00014",
            (
                "What technologies are projects combining to improve performance and "
                "renewable integration?"
            ),
            ["Solar, wind, storage and digitalisation."],
            "irena-hybrid",
        ),
        (
            "irena",
            "element_00069",
            (
                "What role are storage technologies expected to play in "
                "short-duration system flexibility?"
            ),
            [
                (
                    "Storage technologies are expected to provide the majority of "
                    "short-duration flexibility needs."
                )
            ],
            "irena-storage",
        ),
        (
            "irena",
            "element_00071",
            (
                "Which battery chemistry is particularly prominent in utility-scale "
                "battery deployment?"
            ),
            ["Lithium iron phosphate (LFP) chemistry."],
            "irena-storage",
        ),
        (
            "irena",
            "element_00083",
            "Which digital tools does IRENA identify for improving renewable asset operations?",
            [
                (
                    "Predictive maintenance, real-time performance monitoring and "
                    "AI-enabled asset management."
                )
            ],
            "irena-digital",
        ),
        (
            "irena",
            "element_00084",
            "Which technologies support consumer demand response programmes?",
            ["Smart meters, dynamic pricing systems and Internet of Things enabled appliances."],
            "irena-digital",
        ),
        (
            "irena",
            "element_00088",
            "What grid and procurement bottlenecks are delaying wind and solar projects?",
            [
                (
                    "Grid connection bottlenecks and long procurement lead times for "
                    "transformers and high-voltage cables."
                )
            ],
            "irena-grid",
        ),
        (
            "irena",
            "element_00091",
            "How do vertically integrated Chinese renewable supply chains create cost advantages?",
            [
                (
                    "They reduce procurement delays, compress margins and enable "
                    "efficient scaling of gigawatt-scale projects."
                )
            ],
            "irena-supply",
        ),
        (
            "irena",
            "element_00073",
            "What benefits can hybrid solar or wind systems with battery storage provide?",
            ["Firmer output profiles, improved capacity factors and enhanced grid reliability."],
            "irena-hybrid",
        ),
        (
            "imf",
            "element_00009",
            "What oil prices per barrel are assumed for 2025 and 2026 in this WEO appendix?",
            ["2025: USD 68.92 per barrel.", "2026: USD 65.84 per barrel."],
            "imf-oil",
            ["68.92", "65.84"],
        ),
        (
            "imf",
            "element_00024",
            (
                "How are multiyear growth averages generally expressed, and which "
                "EMDE aggregates use geometric averages?"
            ),
            [
                "Multiyear growth averages are compound annual rates of change.",
                (
                    "Inflation and money growth for emerging market and developing "
                    "economies use geometric averages."
                ),
            ],
            "imf-aggregation",
        ),
        (
            "imf",
            "element_00025",
            "How are country-group exchange-rate and interest-rate composites weighted?",
            [
                (
                    "By GDP converted to US dollars at market exchange rates, averaged "
                    "over the preceding three years, as a share of group GDP."
                )
            ],
            "imf-aggregation",
        ),
        (
            "imf",
            "element_00026",
            "What GDP valuation is used to weight other domestic-economy composites?",
            ["GDP valued at purchasing power parity, as a share of world or group GDP."],
            "imf-aggregation",
        ),
        (
            "imf",
            "element_00037",
            "How are composite unemployment rates and employment growth weighted?",
            ["By labor force as a share of group labor force."],
            "imf-aggregation",
        ),
        (
            "imf",
            "element_00040",
            (
                "What minimum representation threshold generally applies when "
                "computing WEO group composites?"
            ),
            ["90 percent or more of the share of group weights."],
            "imf-aggregation",
            ["90"],
        ),
        (
            "imf",
            "element_00041",
            (
                "Do WEO data generally follow calendar years, and where are "
                "exceptional reporting periods listed?"
            ),
            [
                "Data generally refer to calendar years.",
                (
                    "Table F lists economies with exceptional national accounts and "
                    "government finance reporting periods."
                ),
            ],
            "imf-reporting",
        ),
        (
            "imf",
            "element_00044",
            "Why are Afghanistan projections for 2025–30 omitted?",
            [
                (
                    "An unusually high degree of uncertainty, with IMF engagement paused "
                    "because of uncertainty over international recognition of the "
                    "government."
                )
            ],
            "imf-country-notes",
        ),
        (
            "imf",
            "element_00047",
            "Which years of Bolivia projections are omitted, and why?",
            [
                (
                    "Projections for 2026–30 are omitted owing to significant "
                    "uncertainty regarding the economic outlook."
                )
            ],
            "imf-country-notes",
        ),
        (
            "imf",
            "element_00052",
            (
                "Which national accounts base year underlies India real GDP growth "
                "rates in this appendix?"
            ),
            ["2011/12."],
            "imf-country-notes",
            ["2011/12"],
        ),
    ]
    for item in narrative:
        name, eid, q, facts, group, *numbers = item
        add("narrative", q, facts, [source.need(name, eid)], group, numbers[0] if numbers else None)
    # Three source table rows in the original synthetic sample.
    table = source.element("original", "element_00007")
    for row in [1, 2, 3]:
        cells = {c["column"]: c["text"] for c in table["cells"] if c["row"] == row}
        add(
            "table",
            (
                f"For sensor {cells[0]}, report humidity, number of trials, mean "
                f"latency and error rate in Table 1."
            ),
            [
                (
                    f"Humidity {cells[1]}; {cells[2]} trials; mean latency {cells[3]} ms; "
                    f"error rate {cells[4]}%."
                )
            ],
            [source.need("original", table["id"], [(row, c) for c in range(5)])],
            "original-sensors",
            [cells[3], cells[4]],
        )
    # All Table S1/S2 selected values have prior source-page transcriptions.
    table = source.element("irena", "element_00020")
    for row in range(3, 10):
        cells = {c["column"]: c["text"] for c in table["cells"] if c["row"] == row}
        add(
            "table",
            (
                f"In IRENA Table S1, what are the 2024 total installed cost and "
                f"capacity factor for {cells[0]}?"
            ),
            [f"Installed cost: {cells[2]} in 2024 USD/kW; capacity factor: {cells[5]}%."],
            [source.need("irena", table["id"], [(row, c) for c in [0, 2, 5]])],
            "irena-costs",
            [cells[2], cells[5]],
        )
    table = source.element("irena", "element_00122")
    for column in range(1, 5):
        country = next(c["text"] for c in table["cells"] if c["row"] == 0 and c["column"] == column)
        vals = [
            next(c["text"] for c in table["cells"] if c["row"] == row and c["column"] == column)
            for row in [1, 2, 3]
        ]
        add(
            "table",
            (
                f"For {country}, what avoided fuel costs, avoided air-pollution "
                f"damages and total benefits does Table S2 report for 2024?"
            ),
            [
                (
                    f"Avoided fossil fuel costs: USD {vals[0]} billion; avoided air "
                    f"pollution damages: USD {vals[1]} billion; total: USD {vals[2]} "
                    f"billion."
                )
            ],
            [source.need("irena", table["id"], [(row, 0), (row, column)]) for row in [1, 2, 3]]
            + [source.need("irena", table["id"], [(0, column)])],
            "irena-benefits",
            vals,
        )
    # Four Table A rows, with column paths distinguishing group and world shares.
    table = source.element("imf", "element_00111")
    for row in [2, 3, 4, 41]:
        vals = {c["column"]: c["text"] for c in table["cells"] if c["row"] == row}
        add(
            "table",
            (
                f"In IMF Table A, what are the world GDP, world exports and world "
                f"population shares for {vals[0]} in 2024?"
            ),
            [
                (
                    f"World GDP: {vals[3]}%; world exports of goods and services: "
                    f"{vals[5]}%; world population: {vals[7]}%."
                )
            ],
            [source.need("imf", table["id"], [(row, c) for c in [0, 3, 5, 7]])],
            "imf-classification",
            [vals[3], vals[5], vals[7]],
        )
    selections = [("element_00155", r) for r in [2, 3, 4, 41, 42]] + [
        ("element_00195", r) for r in [2, 3, 33, 34]
    ]
    for eid, row in selections:
        table = source.element("imf", eid)
        v = {c["column"]: c["text"] for c in table["cells"] if c["row"] == row}
        add(
            "table",
            (
                f"In Table G, what currency, national-accounts base year and "
                f"national-accounts system are listed for {v[0]}?"
            ),
            [f"Currency: {v[1]}; base year: {v[4]}; system: {v[5]}."],
            [source.need("imf", eid, [(row, c) for c in [0, 1, 4, 5]])],
            "imf-documentation",
            [v[4]],
        )
    for eid, row in [("element_00160", 2), ("element_00160", 3), ("element_00160", 41)]:
        t = source.element("imf", eid)
        v = {c["column"]: c["text"] for c in t["cells"] if c["row"] == row}
        add(
            "table",
            (
                f"What government-finance subsector coverage and accounting-practice "
                f"codes does Table G list for {v[0]}?"
            ),
            [
                (
                    f"Subsector coverage: {v[4]}; accounting practice: {v[5]}. Preserve "
                    f"codes as printed; ellipses indicate missing information."
                )
            ],
            [source.need("imf", eid, [(row, c) for c in [0, 4, 5]])],
            "imf-documentation",
        )
    # Seven source-correlated line-chart endpoint pairs, three grouped bars,
    # three regional solar panels and two numerical-abstention chart cases.
    chart = source.element("irena", "element_00036")
    for series in chart["series"]:
        points = series["points"]
        facts = [f"{p['label']}: {p['value']} {series['unit']}." for p in points]
        add(
            "chart",
            f"What printed 2010 and 2024 LCOE endpoints does Figure S1 show for {series['name']}?",
            facts,
            [
                source.need("irena", chart["id"], series=series["name"], labels=[p["label"]])
                for p in points
            ],
            "irena-costs",
            [str(p["value"]) for p in points],
        )
    chart = source.element("irena", "element_00133")
    series = chart["series"][0]
    for region in ["Africa", "Asia", "Europe"]:
        points = [p for p in series["points"] if p["label"] in [region + " 2023", region + " 2024"]]
        assert len(points) == 2
        add(
            "chart",
            (
                f"What cumulative installed renewable capacity does Figure 1.1 show "
                f"for {region} in 2023 and 2024?"
            ),
            [f"{p['label']}: {p['value']} GW." for p in points],
            [
                source.need("irena", chart["id"], series=series["name"], labels=[p["label"]])
                for p in points
            ],
            "irena-capacity",
            [str(p["value"]) for p in points],
        )
    chart = source.element("irena", "element_00044")
    for region in ["China", "India", "Brazil"]:
        series = next(s for s in chart["series"] if s["name"] == region)
        points = series["points"]
        add(
            "chart",
            (
                f"What printed solar-PV LCOE values are shown for {region} in Figure "
                f"S2 from 2022 through 2024?"
            ),
            [f"{p['label']}: {p['value']} USD/kWh." for p in points],
            [source.need("irena", chart["id"], series=region, labels=[p["label"]]) for p in points],
            "irena-costs",
            [str(p["value"]) for p in points],
        )
    for eid, figure, question, fact in [
        (
            "element_00079",
            "S6",
            (
                "Can exact individual project coordinates be read from the printed "
                "point labels in Figure S6?"
            ),
            (
                "No exact per-project numeric labels are provided; axis ticks must "
                "not be treated as project observations."
            ),
        ),
        (
            "element_00099",
            "S7",
            (
                "Does Figure S7 provide printed exact values for every individual "
                "WACC scatter observation?"
            ),
            (
                "No exact per-observation numeric labels are provided; exact "
                "individual values should not be invented."
            ),
        ),
    ]:
        add(
            "chart",
            question,
            [fact],
            [source.need("irena", eid)],
            "irena-scatter-" + figure,
            status="Source-reviewed scatter abstention criterion",
        )
    images = [
        (
            "element_00002",
            (
                "Describe the renewable technologies and blade markings visible in "
                "the opening IRENA photograph."
            ),
            "A white wind turbine with red blade markings, with solar panels in the foreground.",
        ),
        (
            "element_00040",
            "Describe the setting of the solar-panel photograph with the city skyline.",
            (
                "Rows of solar panels stand above water, with a dense urban skyline "
                "behind them. The photograph alone does not establish whether the "
                "panels float."
            ),
        ),
        (
            "element_00118",
            (
                "Describe the visible equipment and lifting arrangement in the "
                "turbine nacelle close-up."
            ),
            (
                "An open white turbine nacelle, a large suspended circular "
                "component, crane cables and a visible person. A still image does "
                "not establish the direction of motion."
            ),
        ),
        (
            "recovered_image_00001",
            (
                "What infrastructure and surrounding landscape are visible on the "
                "Latest Cost Trends chapter-divider photograph?"
            ),
            (
                "A large electrical transmission tower and overhead lines, other "
                "towers in the distance, a green field and sky."
            ),
        ),
        (
            "element_00183",
            "Describe the structures and water flow visible in the hydroelectric dam photograph.",
            (
                "A concrete dam with steep spillways, flowing water and white spray, "
                "orange structures at the top, buildings and mountainous "
                "surroundings."
            ),
        ),
    ]
    for eid, q, fact in images:
        add(
            "image",
            q,
            [fact],
            [source.need("irena", eid, text=fact)],
            "image-" + eid,
            status="Assistant inspected the actual source crop; not independent human adjudication",
        )
    # Fifteen genuine multi-page Table G joins, preserving codes rather than
    # assuming a code expansion that the excerpt does not establish.
    for row in [2, 3, 4, 41, 42]:
        a = source.element("imf", "element_00155")
        b = source.element("imf", "element_00160")
        av = {c["column"]: c["text"] for c in a["cells"] if c["row"] == row}
        bv = {c["column"]: c["text"] for c in b["cells"] if c["row"] == row}
        pairs = [
            (
                [1],
                [3],
                (
                    f"For {av[0]}, combine the currency and government-finance "
                    f"statistics-manual code from the two Table G page groups."
                ),
                f"Currency: {av[1]}; government-finance statistics-manual code: {bv[3]}.",
            ),
            (
                [3],
                [7],
                (
                    f"For {av[0]}, compare the latest actual annual national-accounts and "
                    f"balance-of-payments periods in Table G."
                ),
                f"National accounts: {av[3]}; balance of payments: {bv[7]}.",
            ),
            (
                [2],
                [6],
                (
                    f"For {av[0]}, which source codes are listed for national accounts "
                    f"and balance of payments across the continued Table G pages?"
                ),
                (
                    f"National accounts historical source: {av[2]}; balance of payments "
                    f"historical source: {bv[6]}."
                ),
            ),
        ]
        for ac, bc, q, fact in pairs:
            add(
                "cross_page",
                q,
                [fact],
                [
                    source.need("imf", a["id"], [(row, c) for c in [0, *ac]]),
                    source.need("imf", b["id"], [(row, c) for c in [0, *bc]]),
                ],
                "imf-documentation",
            )
    table = source.element("original", "element_00007")
    for row in [1, 2, 3]:
        v = {c["column"]: c["text"] for c in table["cells"] if c["row"] == row}
        flag = float(v[4]) > 3
        branch = (
            "true, so the reading is flagged"
            if flag
            else "false, so this branch does not flag the reading"
        )
        add(
            "cross_page",
            (
                f"Using sensor {v[0]}'s Table 1 error rate and the page-2 calibration "
                f"flowchart, should the error-greater-than-3-percent branch flag the "
                f"reading?"
            ),
            [(f"The error rate is {v[4]}%. The greater-than-3% condition is {branch}.")],
            [
                source.need("original", table["id"], [(row, 0), (row, 4)]),
                source.need("original", "element_00024"),
            ],
            "original-sensors",
            [v[4]],
        )
    add(
        "cross_page",
        (
            "Report the 2024 global installed renewable capacity and the total "
            "avoided fossil fuel costs stated in the IRENA highlights."
        ),
        [
            "Global installed renewable capacity: 4 443 GW.",
            "Avoided fossil fuel costs: USD 467 billion.",
        ],
        [source.need("irena", "element_00023"), source.need("irena", "element_00015")],
        "irena-benefits-capacity",
        ["4443", "467"],
    )
    add(
        "cross_page",
        (
            "Combine the 2024 global installed renewable capacity with China "
            "avoided fossil fuel costs from Table S2."
        ),
        [
            "Global installed renewable capacity: 4 443 GW.",
            "China avoided fossil fuel costs: USD 179.8 billion.",
        ],
        [
            source.need("irena", "element_00023"),
            source.need("irena", "element_00122", [(1, 0), (1, 1)]),
            source.need("irena", "element_00122", [(0, 1)]),
        ],
        "irena-benefits-capacity",
        ["4443", "179.8"],
    )
    absent = [
        (
            "What exact solar-PV LCOE does the provided IRENA selection project for 2035?",
            "No 2035 solar-PV LCOE projection is provided.",
        ),
        (
            "What exact onshore-wind installed cost does this selection forecast for 2035?",
            "No 2035 onshore-wind installed-cost forecast is provided.",
        ),
        (
            (
                "What is the exact Pearson correlation coefficient for the project "
                "observations in Figure S6?"
            ),
            "The provided figure and text do not supply an exact Pearson correlation coefficient.",
        ),
        (
            "List exact x and y coordinates for every individual project in Figure S6.",
            "The scatter observations do not have exact printed per-project coordinate labels.",
        ),
        (
            (
                "What manufacturer model and serial number identify the turbine in "
                "the nacelle photograph?"
            ),
            (
                "The provided photograph and description do not establish the "
                "manufacturer model and serial number."
            ),
        ),
        (
            "What are the exact GPS coordinates of the dam in the IRENA photograph?",
            "Exact GPS coordinates are not provided.",
        ),
        (
            "What oil price per barrel does this WEO appendix assume for 2035?",
            "The provided appendix does not give a 2035 oil-price assumption.",
        ),
        (
            (
                "What is Albania quarterly GDP-growth forecast for the first quarter "
                "of 2030 in the supplied appendix?"
            ),
            (
                "The supplied documentation and classification tables do not provide "
                "that quarterly forecast."
            ),
        ),
        (
            (
                "What is the 95-percent confidence interval around the stated 2025 "
                "oil-price assumption?"
            ),
            "No 95-percent confidence interval is supplied for that assumption.",
        ),
        (
            "List all twelve individual latency observations used to compute the BY-04 table mean.",
            (
                "The synthetic table gives summary statistics, not the twelve "
                "individual trial observations."
            ),
        ),
    ]
    for i, (q, fact) in enumerate(absent):
        add(
            "unanswerable",
            q,
            [fact],
            [],
            f"absence-{i}",
            status=(
                "Scope-limited absence reviewed against supplied selections; not a "
                "claim about all external documents"
            ),
        )
    # Pre-scoring source review adds an explicit footnote, a source caveat,
    # and a deterministic cross-page arithmetic case, without changing quotas.
    q = next(q for q in result if q["id"] == "q049")
    q.update(
        question=(
            "Does Table S2 include avoided carbon-dioxide emissions in its total "
            "benefits, and what additional climate benefit does footnote 13 "
            "estimate?"
        ),
        reference=(
            "Avoided carbon-dioxide emissions are excluded from the listed "
            "total. At USD 190 per tonne, 597 MtCO2 avoided in 2024 implies an "
            "additional USD 113.5 billion in monetised climate benefits."
        ),
        required_facts=[
            "CO2 benefits are excluded from the listed total.",
            ("The footnote uses USD 190 per tonne, 597 MtCO2 and an additional USD 113.5 billion."),
        ],
        evidence=[source.need("irena", "element_00124")],
        group_id="irena-benefits",
        numeric_values=["190", "597", "113.5"],
    )
    q = next(q for q in result if q["id"] == "q082")
    q.update(
        question=(
            "What does the WEO caveat say about treating 2024 figures as actual "
            "outturns, and what latest actual national-accounts year does Table "
            "G list for the Democratic Republic of the Congo?"
        ),
        reference=(
            "For some countries, 2024 and earlier figures are estimates rather "
            "than actual outturns. Table G lists 2020 as the latest actual "
            "annual national-accounts data for the Democratic Republic of the "
            "Congo."
        ),
        required_facts=[
            "Some 2024 and earlier figures are estimates, not actual outturns.",
            ("The Democratic Republic of the Congo national-accounts latest actual year is 2020."),
        ],
        evidence=[
            source.need("imf", "element_00042"),
            source.need("imf", "element_00155", [(41, 0), (41, 3)]),
        ],
        numeric_values=["2024", "2020"],
    )
    q = next(q for q in result if q["id"] == "q090")
    q.update(
        question=(
            "What percentage of global avoided fossil-fuel costs in 2024 does "
            "China account for using Table S2 and the IRENA highlights? State "
            "the two amounts and round the percentage to one decimal place."
        ),
        reference=(
            "China accounts for USD 179.8 billion out of a global USD 467 "
            "billion, or approximately 38.5%."
        ),
        required_facts=[
            "China avoided fossil-fuel costs: USD 179.8 billion.",
            "Global avoided fossil-fuel costs: USD 467 billion.",
            "179.8 / 467 * 100 rounds to 38.5%.",
        ],
        evidence=[
            source.need("irena", "element_00015"),
            source.need("irena", "element_00122", [(1, 0), (1, 1)]),
            source.need("irena", "element_00122", [(0, 1)]),
        ],
        numeric_values=["179.8", "467", "38.5"],
        derivation=dict(
            operation="179.8 / 467 * 100",
            rounding_decimal_places=1,
            expected="38.5",
            unit="percent",
        ),
    )
    assert Counter(q["category"] for q in result) == Counter(
        narrative=20, table=30, chart=15, image=5, cross_page=20, unanswerable=10
    ), Counter(q["category"] for q in result)
    return result


def assign_splits(rows):
    # Union every shared source element, then explicit semantic families. This
    # keeps repeated table/chart facts and related cross-page joins together.
    roots = {q["id"]: q["id"] for q in rows}

    def root(x):
        while roots[x] != x:
            x = roots[x]
        return x

    def union(a, b):
        roots[root(b)] = root(a)

    seen = {}
    for q in rows:
        keys = [("family", q["group_id"])] + [(e.document_id, e.element_id) for e in q["evidence"]]
        for key in keys:
            if key in seen:
                union(q["id"], seen[key])
            else:
                seen[key] = q["id"]
    groups = {}
    for q in rows:
        groups.setdefault(root(q["id"]), []).append(q)
    ordered = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(("rag100-v1:" + group[0]["id"]).encode()).hexdigest(),
    )
    categories = list(Counter(q["category"] for q in rows))
    bit = {name: 1 << i for i, name in enumerate(categories)}
    choices = {(0, 0): []}
    for i, group in enumerate(ordered):
        mask = 0
        for q in group:
            mask |= bit[q["category"]]
        for (total, covered), indices in list(choices.items())[::-1]:
            target = (total + len(group), covered | mask)
            if target[0] <= 30 and target not in choices:
                choices[target] = indices + [i]
    target = (30, (1 << len(categories)) - 1)
    if target not in choices:
        raise ValueError(
            "Cannot allocate a balanced 30-question development set without splitting source groups"
        )
    selected = choices[target]

    for i, group in enumerate(ordered):
        group_id = (
            "group-" + hashlib.sha256("|".join(q["id"] for q in group).encode()).hexdigest()[:12]
        )
        for q in group:
            q.update(group_id=group_id, split="development" if i in selected else "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    sources = Sources()
    rows = make_questions(sources)
    assign_splits(rows)
    questions = [BenchmarkQuestion.model_validate(q) for q in rows]
    validation = validate_dataset(questions)
    target = args.output_directory / "questions.jsonl"
    target.write_text("".join(q.model_dump_json() + "\n" for q in questions))
    with (args.output_directory / "questions.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["id", "category", "split", "question", "reference", "group_id"]
        )
        writer.writeheader()
        writer.writerows({k: q.model_dump()[k] for k in writer.fieldnames} for q in questions)
    save(
        args.output_directory / "expected-evidence.json",
        {q.id: [e.model_dump() for e in q.evidence] for q in questions},
    )
    save(
        args.output_directory / "manifest.json",
        dict(
            benchmark_id="ragas-100-v1",
            questions_sha256=digest(target),
            validation=validation,
            index_manifest_sha256=digest(
                Path("outputs/rag_indexes/bge-large-hybrid-02/index_manifest.json")
            ),
            reference_origin=(
                "Source-compiled silver references; GPT-OSS drafts will be preserved "
                "separately before evaluation."
            ),
            split_policy=(
                "Semantic families and shared source elements are indivisible; "
                "deterministic group subset yields 30 development and 70 test. "
                "Documents were previously used in development; not an "
                "unseen-document benchmark."
            ),
        ),
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
