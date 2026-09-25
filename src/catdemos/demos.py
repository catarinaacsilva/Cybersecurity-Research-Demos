"""Registry of the demo servers: module, default port and what the hub shows for each."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Demo:
    """One demo server as the hub presents it."""

    key: str
    title: str
    port: int
    pitch: str
    interaction: str
    papers: tuple[str, ...]


DEMOS: tuple[Demo, ...] = (
    Demo(
        "asap",
        "ASAP Latent Map",
        8001,
        "Can a privacy-harmful Android app be spotted from the permissions it asks for?",
        "Move the mouse over a map of 29,332 real apps; drag the detection threshold.",
        (
            "ASAP: A Dynamic & Proactive Approach for Android Security Analysis and Privacy (2024)",
            "ASAP 2.0: Autonomous & Proactive Detection of Malicious Applications for Privacy "
            "Quantification in 6G Network Services (Computer Communications, 2025)",
            "Assessing Mobile Application Privacy: A Quantitative Framework for Privacy Measurement (2023)",
        ),
    ),
    Demo(
        "psdc",
        "PsDC Categorizer",
        8002,
        "How does PsDC categorize the data an app collects, and what is that data worth?",
        "Drag fields into an app's data model; set how it is collected, kept and how precise it is.",
        (
            "Proactive Data Categorization for Privacy in DevPrivOps (Information, 2025)",
            "Semantic and Numerical Feature Clustering for Automated Privacy Quantification (FiCloud, 2025)",
        ),
    ),
    Demo(
        "dp",
        "DP vs Profiling",
        8003,
        "Differential privacy stops re-identification, but does it stop profiling?",
        "Choose which fields to hide with differential privacy; move along the privacy budget ε.",
        ("Evaluating the Effectiveness of Differential Privacy Against Profiling (ICCT-Europe, 2025)",),
    ),
    Demo(
        "elastic",
        "Elastic Privacy Map",
        8004,
        "Can a phone hide a patient's whereabouts every day, yet stay precise during a crisis?",
        "Draw a walk around Fábrica and the UA campus; pause or scribble to act out a crisis.",
        ("Self-Adaptive Governance for Elastic Privacy in Mental Health Digital Phenotyping (2026)",),
    ),
)
HUB_PORT = 8000
