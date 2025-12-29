from typing import Optional
from src.models.session import SessionManager
from src.agent.planner import Planner
from src.agent.decision import DecisionMaker
from src.research.search import WebSearcher
from src.research.extractor import ContentExtractor
from src.research.claims import ClaimExtractor
from src.research.ranker import SourceRanker
from src.analysis.gaps import GapDetector
from src.analysis.clarifier import Clarifier

_session_manager = SessionManager()
_planner: Optional[Planner] = None
_searcher: Optional[WebSearcher] = None
_extractor: Optional[ContentExtractor] = None
_claim_extractor: Optional[ClaimExtractor] = None
_ranker: Optional[SourceRanker] = None
_gap_detector: Optional[GapDetector] = None
_clarifier: Optional[Clarifier] = None
_decision_maker: Optional[DecisionMaker] = None


def _get_planner():
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner


def _get_searcher():
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher()
    return _searcher


def _get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = ContentExtractor()
    return _extractor


def _get_claim_extractor():
    global _claim_extractor
    if _claim_extractor is None:
        _claim_extractor = ClaimExtractor()
    return _claim_extractor


def _get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = SourceRanker()
    return _ranker


def _get_gap_detector():
    global _gap_detector
    if _gap_detector is None:
        _gap_detector = GapDetector()
    return _gap_detector


def _get_clarifier():
    global _clarifier
    if _clarifier is None:
        _clarifier = Clarifier()
    return _clarifier


def _get_decision_maker():
    global _decision_maker
    if _decision_maker is None:
        _decision_maker = DecisionMaker()
    return _decision_maker


def run_full_pipeline(question: str) -> dict:
    session = _session_manager.create_session(question)
    
    planner = _get_planner()
    decomposition = planner.decompose(question)
    sub_questions = decomposition.sub_questions
    
    searcher = _get_searcher()
    discovery = searcher.search_all(sub_questions)
    sources = discovery.all_sources
    
    extractor = _get_extractor()
    extraction = extractor.extract_all(sources)
    sources_with_content = [
        r.source for r in extraction.results 
        if r.success and r.content_length > 100
    ]
    
    claim_extractor = _get_claim_extractor()
    evidence = claim_extractor.extract_all(sources_with_content[:10])
    
    ranker = _get_ranker()
    claims_by_source = {}
    for sc in evidence.claims_by_source:
        claims_by_source[sc.source_url] = sc.claims
    ranking = ranker.rank_sources(sources_with_content[:10], claims_by_source)
    
    gap_detector = _get_gap_detector()
    gaps = gap_detector.analyze(question, evidence)
    
    clarifier = _get_clarifier()
    clarification = clarifier.generate_questions(question, gaps)
    
    decision_maker = _get_decision_maker()
    recommendation = decision_maker.make_recommendation(
        question=question,
        evidence=evidence,
        gaps=gaps,
        ranking=ranking,
    )
    
    return {
        "question": question,
        "session_id": session.session_id,
        "recommendation": {
            "confidence": recommendation.confidence,
            "decision": recommendation.decision,
            "key_reasons": recommendation.key_reasons,
            "trade_offs": recommendation.trade_offs,
            "risks": recommendation.risks,
            "next_steps": recommendation.next_steps,
            "disclaimer": recommendation.disclaimer,
        },
        "reasoning": {
            "gaps": {
                "unknowns": [
                    {"description": u.description, "importance": u.importance}
                    for u in gaps.unknowns[:5]
                ],
                "conflicts": [
                    {"description": c.description}
                    for c in gaps.conflicts[:3]
                ],
                "assumptions": [
                    {"description": a.description}
                    for a in gaps.assumptions[:3]
                ],
            },
            "top_sources": [
                {
                    "title": score.source.title,
                    "domain": score.source.domain,
                    "score": round(score.total_score, 2),
                }
                for score in ranking.top_3
            ],
            "clarifying_questions": [
                {
                    "question": q.question,
                    "why": q.why_it_matters,
                }
                for q in clarification.questions[:5]
            ],
        },
        "stats": {
            "sources_analyzed": len(sources_with_content),
            "claims_extracted": evidence.total_claims,
            "gaps_identified": gaps.total_gaps,
        }
    }
