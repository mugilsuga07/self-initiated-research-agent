import sys
import os

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.models.session import SessionManager, InputValidationError
from src.models.source import DiscoveryResults
from src.models.claim import EvidenceSummary
from src.agent.planner import Planner, DecompositionResult
from src.research.search import WebSearcher
from src.research.extractor import ContentExtractor, ExtractionSummary
from src.research.claims import ClaimExtractor
from src.research.ranker import SourceRanker, RankingResult
from src.analysis.gaps import GapDetector, GapAnalysisResult
from src.analysis.clarifier import Clarifier, ClarificationRequest
from src.agent.decision import DecisionMaker, Recommendation

console = Console()
session_manager = SessionManager()

_planner = None
_searcher = None
_extractor = None
_claim_extractor = None
_ranker = None
_gap_detector = None
_clarifier = None
_decision_maker = None


def get_planner() -> Planner:
    global _planner
    if _planner is None:
        _planner = Planner()
    return _planner


def get_searcher() -> WebSearcher:
    global _searcher
    if _searcher is None:
        _searcher = WebSearcher()
    return _searcher


def get_extractor() -> ContentExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ContentExtractor()
    return _extractor


def get_claim_extractor() -> ClaimExtractor:
    global _claim_extractor
    if _claim_extractor is None:
        _claim_extractor = ClaimExtractor()
    return _claim_extractor


def get_ranker() -> SourceRanker:
    global _ranker
    if _ranker is None:
        _ranker = SourceRanker()
    return _ranker


def get_gap_detector() -> GapDetector:
    global _gap_detector
    if _gap_detector is None:
        _gap_detector = GapDetector()
    return _gap_detector


def get_clarifier() -> Clarifier:
    global _clarifier
    if _clarifier is None:
        _clarifier = Clarifier()
    return _clarifier


def get_decision_maker() -> DecisionMaker:
    global _decision_maker
    if _decision_maker is None:
        _decision_maker = DecisionMaker()
    return _decision_maker


def display_session(session) -> None:
    console.print()
    console.print(Panel(
        f"[bold cyan]Session:[/bold cyan] {session.session_id}\n"
        f"[bold green]Question:[/bold green] {session.question}",
        title="Step 1: Session Created",
        border_style="green"
    ))


def display_sub_questions(result: DecompositionResult) -> None:
    lines = []
    for i, sq in enumerate(result.sub_questions, 1):
        lines.append(f"[cyan]{i}.[/cyan] {sq}")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title=f"Step 2: Sub-Questions ({len(result.sub_questions)} generated)",
        border_style="blue"
    ))


def display_discovery_results(results: DiscoveryResults, using_mock: bool = False) -> None:
    lines = []
    
    if using_mock:
        lines.append("[yellow]Using mock data (no search API key configured)[/yellow]")
        lines.append("")
    
    lines.append(f"[bold]Total sources:[/bold] {results.total_sources}")
    lines.append(f"[bold]Unique domains:[/bold] {len(results.unique_domains)}")
    lines.append("")
    
    for sq, search_results in results.results_by_question.items():
        sq_display = sq[:60] + "..." if len(sq) > 60 else sq
        lines.append(f"[cyan]Q:[/cyan] {sq_display}")
        lines.append(f"   -> {len(search_results.sources)} sources")
        
        for source in search_results.sources[:2]:
            title_short = source.title[:50] + "..." if len(source.title) > 50 else source.title
            lines.append(f"   [dim]- {title_short}[/dim]")
        
        if len(search_results.sources) > 2:
            lines.append(f"   [dim]  ... and {len(search_results.sources) - 2} more[/dim]")
        lines.append("")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title=f"Step 3: Sources Found ({results.total_sources} total)",
        border_style="magenta"
    ))


def display_extraction_results(summary: ExtractionSummary) -> None:
    lines = []
    lines.append(f"[bold]Extracted:[/bold] {summary.successful}/{summary.total} sources ({summary.success_rate:.0%})")
    if summary.fallback_count > 0:
        lines.append(f"[yellow]Fallbacks:[/yellow] {summary.fallback_count} (used search snippet)")
    lines.append("")
    
    successful_results = [r for r in summary.results if r.success][:5]
    
    for result in successful_results:
        title_short = result.source.title[:45] + "..." if len(result.source.title) > 45 else result.source.title
        status = "[green]+[/green]" if not result.used_fallback else "[yellow]~[/yellow]"
        lines.append(f"{status} [bold]{title_short}[/bold]")
        lines.append(f"   [dim]~{result.content_length:,} chars[/dim]")
        
        preview = result.preview[:150].replace('\n', ' ')
        if len(result.preview) > 150:
            preview += "..."
        lines.append(f"   [dim]\"{preview}\"[/dim]")
        lines.append("")
    
    if len(successful_results) < summary.successful:
        lines.append(f"[dim]... and {summary.successful - len(successful_results)} more extracted[/dim]")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title=f"Step 4: Content Extracted ({summary.successful} sources)",
        border_style="cyan"
    ))


def display_evidence_results(evidence: EvidenceSummary) -> None:
    lines = []
    stats = evidence.summary_stats()
    lines.append(f"[bold]Total claims:[/bold] {stats['total_claims']} from {stats['sources_processed']} sources")
    lines.append(f"[bold]Actionable:[/bold] {evidence.actionable_ratio:.0%} (risks, limitations, practices)")
    lines.append("")
    
    by_type = stats.get('by_type', {})
    type_summary = []
    for claim_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        type_summary.append(f"{claim_type}: {count}")
    
    if type_summary:
        lines.append("[dim]" + " | ".join(type_summary) + "[/dim]")
        lines.append("")
    
    claims_by_type = evidence.claims_by_type()
    priority_types = ['risk', 'failure', 'limitation', 'practice', 'metric', 'example', 'benefit']
    shown_count = 0
    max_to_show = 8
    
    for type_name in priority_types:
        from src.models.claim import ClaimType
        claim_type = ClaimType(type_name) if type_name in [t.value for t in ClaimType] else None
        if not claim_type or claim_type not in claims_by_type:
            continue
        
        claims = claims_by_type[claim_type][:2]
        
        for claim in claims:
            if shown_count >= max_to_show:
                break
            
            claim_text = claim.text[:100] + "..." if len(claim.text) > 100 else claim.text
            lines.append(f"[bold]{type_name.upper()}:[/bold] {claim_text}")
            lines.append(f"   [dim]- {claim.source_title[:40]}...[/dim]")
            shown_count += 1
        
        if shown_count >= max_to_show:
            break
    
    if evidence.total_claims > shown_count:
        lines.append(f"\n[dim]... and {evidence.total_claims - shown_count} more claims[/dim]")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title=f"Step 5: Evidence Extracted ({evidence.total_claims} claims)",
        border_style="yellow"
    ))


def display_ranking_results(ranking: RankingResult) -> None:
    lines = []
    lines.append(f"[bold]Ranked {ranking.total_sources} sources by quality[/bold]")
    lines.append("")
    lines.append("[bold]Top Sources:[/bold]")
    
    for score in ranking.top_3:
        bar_filled = int(score.total_score * 10)
        bar = "=" * bar_filled + "-" * (10 - bar_filled)
        
        title_short = score.source.title[:45] + "..." if len(score.source.title) > 45 else score.source.title
        
        lines.append(f"")
        lines.append(f"[green]#{score.rank}[/green] [bold]{title_short}[/bold]")
        lines.append(f"   [cyan]{bar}[/cyan] {score.total_score:.2f}")
        lines.append(f"   [dim]{score.source.domain} - {score.justification}[/dim]")
    
    lines.append("")
    lines.append("[dim]" + ranking.top_sources_justification + "[/dim]")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title="Step 6: Sources Ranked",
        border_style="green"
    ))


def display_gap_analysis(gaps: GapAnalysisResult) -> None:
    lines = []
    summary = gaps.summary()
    lines.append(f"[bold]Analysis found {summary['total_gaps']} gaps[/bold]")
    if summary['has_critical']:
        lines.append("[red]Critical gaps identified[/red]")
    lines.append("")
    
    if gaps.unknowns:
        lines.append("[bold red]UNKNOWNS (Missing Information)[/bold red]")
        for unknown in gaps.unknowns[:5]:
            importance = "[HIGH]" if unknown.importance == "high" else "[MED]" if unknown.importance == "medium" else "[LOW]"
            lines.append(f"   {importance} {unknown.description}")
        if len(gaps.unknowns) > 5:
            lines.append(f"   [dim]... and {len(gaps.unknowns) - 5} more[/dim]")
        lines.append("")
    
    if gaps.conflicts:
        lines.append("[bold yellow]CONFLICTS (Sources Disagree)[/bold yellow]")
        for conflict in gaps.conflicts[:3]:
            lines.append(f"   - {conflict.description}")
            if conflict.source_a and conflict.source_b:
                lines.append(f"     [dim]{conflict.source_a} vs {conflict.source_b}[/dim]")
        if len(gaps.conflicts) > 3:
            lines.append(f"   [dim]... and {len(gaps.conflicts) - 3} more[/dim]")
        lines.append("")
    
    if gaps.assumptions:
        lines.append("[bold cyan]ASSUMPTIONS (Implicit)[/bold cyan]")
        for assumption in gaps.assumptions[:4]:
            lines.append(f"   - {assumption.description}")
            if assumption.risk:
                lines.append(f"     [dim]Risk: {assumption.risk}[/dim]")
        if len(gaps.assumptions) > 4:
            lines.append(f"   [dim]... and {len(gaps.assumptions) - 4} more[/dim]")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title="Step 7: Gap Analysis",
        border_style="red"
    ))


def display_clarifying_questions(clarification: ClarificationRequest) -> None:
    if not clarification.questions:
        return
    
    lines = []
    lines.append(f"[dim]{clarification.context}[/dim]")
    lines.append("")
    
    for q in clarification.questions:
        priority = "[!]" if q.priority == 1 else "[?]"
        lines.append(f"{priority} [bold]{q.question}[/bold]")
        lines.append(f"   [dim]Why: {q.why_it_matters}[/dim]")
        
        if q.example_answers:
            examples = " | ".join(q.example_answers[:3])
            lines.append(f"   [cyan]Examples: {examples}[/cyan]")
        lines.append("")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title=f"Step 8: Clarifying Questions ({len(clarification.questions)})",
        border_style="magenta"
    ))


def display_recommendation(recommendation: Recommendation) -> None:
    lines = []
    
    conf = recommendation.confidence.upper()
    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(conf, "white")
    lines.append(f"[{conf_color}]Confidence: {conf}[/{conf_color}]")
    lines.append("")
    lines.append(f"[bold white]{recommendation.decision}[/bold white]")
    lines.append("")
    
    if recommendation.key_reasons:
        lines.append("[bold cyan]Key Reasons:[/bold cyan]")
        for reason in recommendation.key_reasons:
            lines.append(f"  - {reason}")
        lines.append("")
    
    if recommendation.trade_offs:
        lines.append("[bold yellow]Trade-offs:[/bold yellow]")
        for to in recommendation.trade_offs:
            if isinstance(to, dict):
                pro = to.get("pro", "")
                con = to.get("con", "")
                lines.append(f"  + [green]{pro}[/green]")
                lines.append(f"  - [red]{con}[/red]")
        lines.append("")
    
    if recommendation.risks:
        lines.append("[bold red]Risks & Limitations:[/bold red]")
        for risk in recommendation.risks:
            lines.append(f"  - {risk}")
        lines.append("")
    
    if recommendation.next_steps:
        lines.append("[bold green]Next Steps:[/bold green]")
        for i, step in enumerate(recommendation.next_steps, 1):
            lines.append(f"  {i}. {step}")
        lines.append("")
    
    if recommendation.top_sources:
        lines.append("[bold magenta]Top Sources:[/bold magenta]")
        for src in recommendation.top_sources[:3]:
            title = src.get("title", "Unknown")[:50]
            url = src.get("url", "")
            lines.append(f"  - {title}")
            lines.append(f"    [dim]{url}[/dim]")
        lines.append("")
    
    lines.append(f"[dim italic]{recommendation.disclaimer}[/dim italic]")
    
    content = "\n".join(lines)
    console.print()
    console.print(Panel(
        content,
        title="Step 9: Final Recommendation",
        border_style="bold green"
    ))


def display_error(message: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold red]{message}[/bold red]",
        title="Error",
        border_style="red"
    ))
    console.print()


def display_all_sessions() -> None:
    sessions = session_manager.list_sessions()
    
    if not sessions:
        console.print("[dim]No sessions yet.[/dim]")
        return
    
    table = Table(title="All Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("Question", style="white", max_width=50)
    table.add_column("Sub-Qs", style="yellow", justify="center")
    table.add_column("Sources", style="magenta", justify="center")
    table.add_column("Created", style="dim")
    
    for session in sessions:
        table.add_row(
            session.session_id,
            session.question[:50] + "..." if len(session.question) > 50 else session.question,
            str(len(session.sub_questions)) if session.sub_questions else "-",
            str(len(session.sources)) if session.sources else "-",
            session.created_at.strftime("%H:%M:%S")
        )
    
    console.print(table)


def check_api_key() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        console.print(Panel(
            "[yellow]OpenAI API key not configured.[/yellow]\n\n"
            "To enable Step 2 (Goal Decomposition), set your API key:\n\n"
            "[cyan]export OPENAI_API_KEY='your-key-here'[/cyan]\n\n"
            "Step 1 (Session Setup) will still work.",
            title="API Key Required",
            border_style="yellow"
        ))
        return False
    return True


def run_research_pipeline(question: str) -> None:
    session = session_manager.create_session(question)
    display_session(session)
    
    sub_questions = []
    if check_api_key():
        console.print("[dim]Decomposing question into sub-questions...[/dim]")
        
        try:
            planner = get_planner()
            result = planner.decompose(question)
            session.sub_questions = result.sub_questions
            sub_questions = result.sub_questions
            display_sub_questions(result)
        except Exception as e:
            display_error(f"Goal decomposition failed: {str(e)}")
            return
    else:
        console.print("[dim]Using demo sub-questions for Step 3...[/dim]")
        sub_questions = [
            "How are AI agents currently used in production today?",
            "What failures are commonly reported with AI agents?",
            "What guardrails do engineering teams use for AI agents?",
            "What evidence shows AI agents succeeding at scale?",
        ]
        session.sub_questions = sub_questions
    
    sources = []
    if sub_questions:
        console.print("[dim]Searching for sources...[/dim]")
        
        try:
            searcher = get_searcher()
            discovery_results = searcher.search_all(sub_questions)
            session.sources = discovery_results.all_sources
            sources = discovery_results.all_sources
            display_discovery_results(discovery_results, using_mock=searcher.is_using_mock)
        except Exception as e:
            display_error(f"Source discovery failed: {str(e)}")
            return
    
    sources_with_content = []
    if sources:
        console.print("[dim]Extracting content from sources...[/dim]")
        
        try:
            extractor = get_extractor()
            extraction_summary = extractor.extract_all(sources)
            sources_with_content = [
                r.source for r in extraction_summary.results 
                if r.success and r.content_length > 100
            ]
            display_extraction_results(extraction_summary)
        except Exception as e:
            display_error(f"Content extraction failed: {str(e)}")
            return
    
    evidence = None
    if sources_with_content:
        console.print("[dim]Extracting evidence claims from content...[/dim]")
        
        try:
            claim_extractor = get_claim_extractor()
            evidence = claim_extractor.extract_all(sources_with_content[:10])
            session.claims = evidence.all_claims
            display_evidence_results(evidence)
        except Exception as e:
            display_error(f"Evidence extraction failed: {str(e)}")
    
    ranking = None
    if sources_with_content and evidence:
        console.print("[dim]Ranking sources by quality...[/dim]")
        
        try:
            ranker = get_ranker()
            claims_by_source = {}
            for source_claims in evidence.claims_by_source:
                claims_by_source[source_claims.source_url] = source_claims.claims
            ranking = ranker.rank_sources(sources_with_content[:10], claims_by_source)
            display_ranking_results(ranking)
        except Exception as e:
            display_error(f"Source ranking failed: {str(e)}")
    
    gaps = None
    if evidence and evidence.total_claims > 0:
        console.print("[dim]Analyzing gaps in evidence...[/dim]")
        
        try:
            gap_detector = get_gap_detector()
            gaps = gap_detector.analyze(question, evidence)
            session.gaps = [u.description for u in gaps.unknowns]
            display_gap_analysis(gaps)
        except Exception as e:
            display_error(f"Gap analysis failed: {str(e)}")
    
    if gaps and gaps.total_gaps > 0:
        console.print("[dim]Generating clarifying questions...[/dim]")
        
        try:
            clarifier = get_clarifier()
            clarification = clarifier.generate_questions(question, gaps)
            session.clarifications = [q.question for q in clarification.questions]
            display_clarifying_questions(clarification)
        except Exception as e:
            display_error(f"Clarification generation failed: {str(e)}")
    
    if evidence and gaps:
        console.print("[dim]Generating final recommendation...[/dim]")
        
        try:
            decision_maker = get_decision_maker()
            recommendation = decision_maker.make_recommendation(
                question=question,
                evidence=evidence,
                gaps=gaps,
                ranking=ranking,
            )
            session.recommendation = recommendation.decision
            display_recommendation(recommendation)
        except Exception as e:
            display_error(f"Recommendation generation failed: {str(e)}")
    
    count = session_manager.session_count()
    console.print(f"[dim]Total sessions: {count}[/dim]")


def run_interactive() -> None:
    console.print(Panel(
        "[bold]Autonomous AI Research & Decision Support Agent[/bold]\n\n"
        "Enter a question to research, or:\n"
        "  - Type [cyan]'list'[/cyan] to see all sessions\n"
        "  - Type [cyan]'quit'[/cyan] to exit",
        title="Research Agent",
        border_style="blue"
    ))
    
    while True:
        try:
            question = Prompt.ask("\n[bold yellow]Your question[/bold yellow]")
            
            if question.lower() == 'quit':
                console.print("\n[dim]Goodbye![/dim]\n")
                break
            
            if question.lower() == 'list':
                display_all_sessions()
                continue
            
            run_research_pipeline(question)
            
        except InputValidationError as e:
            display_error(str(e))
        
        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Goodbye![/dim]\n")
            break


def run_single(question: str) -> None:
    try:
        run_research_pipeline(question)
    except InputValidationError as e:
        display_error(str(e))
        sys.exit(1)


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_single(question)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
