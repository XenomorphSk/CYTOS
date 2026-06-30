# Emergent Ventures Application — GB

## What are you working on?

I am building a mathematical bridge between two fields that do not currently talk to each other: quantum tensor-network theory and computational biology.

The specific project is CYTOS — a pre-registered empirical study testing whether Tree Tensor Networks (TTNs), a model class developed in quantum many-body physics, outperform Graph Neural Networks (GNNs) at modeling gene regulatory network dynamics. The answer, across all 10 tested configurations on the DREAM4 benchmark, is yes — with a 4–6× parameter efficiency advantage and stable long-range correlation capture (0.91–0.95) that GNN baselines cannot match at equal parameter count, degrading further as network size grows.

That finding is already published at https://github.com/XenomorphSk/CYTOS with a full pre-registration document (hypotheses, success criteria, and falsification conditions fixed before any data was seen). But the more interesting result came from pushing the framework further: a quantity derived from the TTN's tensor-network structure — bond entropy, computed via QR canonicalization — predicts which gene communities are most sensitive to perturbation, without ever being trained for it. Spearman ρ=0.66, p=4.4×10⁻³⁵. A GNN cannot compute this quantity at all; it has no bond structure.

This is not an isolated finding. I have been running a parallel project (AMR-NEQ) characterizing noise in IBM quantum processors and testing the holographic Ryu-Takayanagi correspondence on NISQ hardware. In that work, I independently discovered the same mathematical pattern: low-order proxy quantities degenerate when the underlying system has symmetry, and the correction requires moving to a higher-order quantity. The same pathology, the same fix, in quantum hardware and in gene regulatory networks. This is the connection I want to develop: not "physics and biology are related" as a vague metaphor, but a specific, computable, falsifiable claim about when and why tensor-network formalisms capture structure that flat-graph models miss.

## What is your background?

I am 20 years old, self-taught, and have no institutional affiliation. My background is offensive security — I have found multiple zero-day vulnerabilities, including one that earned an honorable mention from Google. I got into physics and machine learning independently, driven by interest in Wheeler's "It from Bit" hypothesis. I have no degree, no lab, no collaborators.

What I have is: two pre-registered projects with real results, one published preprint draft, a reusable Python library (pip-installable, dataset-agnostic), hardware access to IBM quantum processors via the open plan, and a documented record of finding and fixing my own errors in public — including results that falsified my initial hypotheses, reported with the same weight as the confirmatory ones.

The pre-registration document for CYTOS currently has 26 sections, each documenting a decision or correction made during development. Every bug is logged. Every negative result is reported. This is not standard practice even in well-funded academic labs; it is what I consider the minimum bar for trusting your own results.

## Why does this matter?

The standard approach in computational biology is to treat regulatory networks as flat graphs and throw GNNs at them. This works, but it discards information that is already known — the modular, hierarchical organization of regulatory systems. The field has tools from quantum information theory that were specifically designed to exploit this kind of structure efficiently, but these tools have not crossed over.

My results show the crossing is possible and the benefit is real, at least in the benchmark regime. The H3 result (negative: uninformed TTN loses to MLP) is as important as H1/H1b — it tells you exactly when the advantage applies and when it doesn't. This is the kind of mechanistic specificity that makes a result useful rather than just promising.

The longer-term target is to extend this to real biological data and, eventually, to use the TTN's structural quantities — bond entropy, perturbation sensitivity prediction — to guide experimental design. Not "CRISPR to fix cancer" as a pitch, but a specific, concrete question: can a trained TTN identify which regulatory modules are dynamically critical before any wet-lab experiment is run, saving experimental resources by prioritizing interventions that the model predicts will have cross-module effects?

## What would the funding enable?

Three things, in order of priority:

**1. Real biological data at scale.** The current results are on DREAM4 (simulated, 10–100 genes). The next validation requires real experimental data — DREAM5 E. coli at manageable scale, or time-series expression data from a well-characterized system like yeast cell cycle. I have already built the infrastructure (STRING database loader, expression timeseries loader, alignment utilities). The bottleneck is compute time for the 20-seed, parameter-matched comparisons at scale, and access to datasets that require institutional registration I currently cannot obtain independently.

**2. arXiv preprint and community feedback.** The preprint is drafted. Submitting it and engaging with the community's response — which will be skeptical, as it should be — is the fastest way to identify what the next experiment needs to be.

**3. Time.** I am self-funding this work currently. Emergent Ventures funding would allow me to work on this without the constraint of needing to prioritize other income, which is the single largest limitation on the pace of the research.

## What makes you the right person to do this?

I am probably not the right person to run a wet lab, manage a team, or navigate institutional grant systems — I have no training in any of those things. I am possibly the right person to sit at the intersection of quantum information, machine learning, and biology and ask whether the math that works in one domain transfers to another, because I came to all three from the outside and I am not committed to any field's conventional assumptions.

The pre-registration discipline I apply — which I adapted from quantum hardware characterization, where I learned that easy-to-believe correlations are easy to fabricate — is not common in ML-for-biology work. The CYTOS pre-registration document is public, with every decision and correction logged. The negative results are in the paper. I think this is worth funding precisely because it is not the typical pitch: I am not claiming a breakthrough, I am claiming a specific, reproducible, falsifiable finding that opens a tractable research direction, and I am showing my work.

**GitHub:** https://github.com/XenomorphSk/CYTOS
**Twitter/X:** @SkXenomorph
