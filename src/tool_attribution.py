import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ToolContribution(BaseModel):
    """Represents a tool's contribution to the final answer"""
    tool_name: str
    tool_type: str  # e.g., "search", "calculator", "retriever"
    invocation_time: str
    execution_duration: float  # seconds
    input_provided: Dict[str, Any]
    output_generated: str
    contribution_score: float  # 0-1, how much this contributed to final answer
    output_used_in_final: bool
    confidence: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None


class SourceAttribution(BaseModel):
    """Attributes specific information to its source"""
    source_type: str  # "tool", "document", "web", "calculation"
    source_identifier: str  # tool name, URL, document path, etc.
    information_extracted: str
    relevance_score: float
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolAttributionReport(BaseModel):
    """Complete attribution report for a query"""
    report_id: str
    query: str
    final_answer: str
    timestamp: str
    tools_used: List[ToolContribution]
    source_attributions: List[SourceAttribution]
    tool_usage_summary: Dict[str, Any]
    answer_composition: Dict[str, float]  # What % from each source
    total_execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolAttributionTracker:
    """
    Tracks tool usage and attributes contributions to the final answer.
    Provides transparency about which tools and sources contributed what information.
    """
    
    def __init__(self, attribution_dir: str = "traces/tool_attribution"):
        self.attribution_dir = Path(attribution_dir)
        self.attribution_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_report: Optional[ToolAttributionReport] = None
        self.tool_contributions: List[ToolContribution] = []
        self.source_attributions: List[SourceAttribution] = []
        self.start_time: Optional[datetime] = None
    
    def start_tracking(self, query: str, metadata: Dict[str, Any] = None) -> str:
        """Initialize tracking for a new query"""
        report_id = f"attr_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.current_report = ToolAttributionReport(
            report_id=report_id,
            query=query,
            final_answer="",  # Will be set later
            timestamp=datetime.now().isoformat(),
            tools_used=[],
            source_attributions=[],
            tool_usage_summary={},
            answer_composition={},
            total_execution_time=0.0,
            metadata=metadata or {}
        )
        
        self.tool_contributions = []
        self.source_attributions = []
        self.start_time = datetime.now()
        
        logger.info(f"Started tool attribution tracking {report_id}")
        return report_id
    
    def track_tool_usage(
        self,
        tool_name: str,
        tool_type: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        execution_duration: float,
        contribution_score: float = 0.5,
        output_used: bool = True,
        confidence: Optional[float] = None,
        error: Optional[str] = None
    ) -> str:
        """Track a tool's usage and contribution"""
        if not self.current_report:
            raise ValueError("No active tracking. Call start_tracking() first.")
        
        contribution = ToolContribution(
            tool_name=tool_name,
            tool_type=tool_type,
            invocation_time=datetime.now().isoformat(),
            execution_duration=execution_duration,
            input_provided=tool_input,
            output_generated=str(tool_output)[:1000],  # Truncate long outputs
            contribution_score=contribution_score,
            output_used_in_final=output_used,
            confidence=confidence,
            error_occurred=error is not None,
            error_message=error
        )
        
        self.tool_contributions.append(contribution)
        logger.debug(f"Tracked tool usage: {tool_name}")
        
        return contribution.tool_name
    
    def add_source_attribution(
        self,
        source_type: str,
        source_identifier: str,
        information: str,
        relevance_score: float,
        metadata: Dict[str, Any] = None
    ):
        """Add attribution for a specific source of information"""
        if not self.current_report:
            raise ValueError("No active tracking. Call start_tracking() first.")
        
        attribution = SourceAttribution(
            source_type=source_type,
            source_identifier=source_identifier,
            information_extracted=information[:500],  # Truncate
            relevance_score=relevance_score,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.source_attributions.append(attribution)
        logger.debug(f"Added source attribution: {source_type} - {source_identifier}")
    
    def calculate_answer_composition(self, final_answer: str) -> Dict[str, float]:
        """
        Analyze the final answer to determine what percentage came from each source.
        This uses contribution scores and output usage.
        """
        if not self.tool_contributions:
            return {}
        
        # Calculate weighted contributions
        total_contribution = 0.0
        tool_contributions_map = defaultdict(float)
        
        for contribution in self.tool_contributions:
            if contribution.output_used_in_final and not contribution.error_occurred:
                weight = contribution.contribution_score
                if contribution.confidence:
                    weight *= contribution.confidence
                
                tool_contributions_map[contribution.tool_name] += weight
                total_contribution += weight
        
        # Normalize to percentages
        if total_contribution > 0:
            composition = {
                tool: (contrib / total_contribution) * 100
                for tool, contrib in tool_contributions_map.items()
            }
        else:
            composition = {}
        
        return composition
    
    def generate_tool_usage_summary(self) -> Dict[str, Any]:
        """Generate summary statistics about tool usage"""
        if not self.tool_contributions:
            return {}
        
        tool_counts = defaultdict(int)
        tool_types = defaultdict(int)
        tool_execution_times = defaultdict(list)
        tool_success_rates = defaultdict(lambda: {"success": 0, "total": 0})
        
        for contribution in self.tool_contributions:
            tool_counts[contribution.tool_name] += 1
            tool_types[contribution.tool_type] += 1
            tool_execution_times[contribution.tool_name].append(contribution.execution_duration)
            
            tool_success_rates[contribution.tool_name]["total"] += 1
            if not contribution.error_occurred:
                tool_success_rates[contribution.tool_name]["success"] += 1
        
        summary = {
            "total_tools_invoked": len(self.tool_contributions),
            "unique_tools_used": len(tool_counts),
            "tool_invocation_counts": dict(tool_counts),
            "tool_type_distribution": dict(tool_types),
            "average_execution_times": {
                tool: sum(times) / len(times)
                for tool, times in tool_execution_times.items()
            },
            "tool_success_rates": {
                tool: rates["success"] / rates["total"] * 100
                for tool, rates in tool_success_rates.items()
            },
            "most_used_tool": max(tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None,
            "total_execution_time": sum(
                c.execution_duration for c in self.tool_contributions
            )
        }
        
        return summary
    
    def finalize_report(self, final_answer: str) -> ToolAttributionReport:
        """Complete the attribution report"""
        if not self.current_report:
            raise ValueError("No active tracking to finalize.")
        
        # Set final answer
        self.current_report.final_answer = final_answer
        
        # Add all tracked contributions
        self.current_report.tools_used = self.tool_contributions
        self.current_report.source_attributions = self.source_attributions
        
        # Calculate composition
        self.current_report.answer_composition = self.calculate_answer_composition(final_answer)
        
        # Generate summary
        self.current_report.tool_usage_summary = self.generate_tool_usage_summary()
        
        # Calculate total time
        if self.start_time:
            end_time = datetime.now()
            self.current_report.total_execution_time = (end_time - self.start_time).total_seconds()
        
        # Save report
        self._save_report()
        
        logger.info(f"Finalized attribution report {self.current_report.report_id}")
        
        report = self.current_report
        self._reset()
        
        return report
    
    def _reset(self):
        """Reset tracker state"""
        self.current_report = None
        self.tool_contributions = []
        self.source_attributions = []
        self.start_time = None
    
    def _save_report(self):
        """Save the attribution report to disk"""
        if not self.current_report:
            return
        
        filename = self.attribution_dir / f"{self.current_report.report_id}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(
                self.current_report.model_dump(), # Changed .dict() to .model_dump()
                f,
                indent=2,
                default=str
            )
        
        logger.info(f"Saved attribution report to {filename}")
    
    def get_report(self, report_id: str) -> ToolAttributionReport:
        """Load a saved attribution report"""
        report_file = self.attribution_dir / f"{report_id}.json"
        
        if not report_file.exists():
            raise ValueError(f"Report {report_id} not found")
        
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            return ToolAttributionReport(**report_data)
    
    def generate_citation_text(self, report_id: str = None) -> str:
        """Generate human-readable citations for the answer"""
        if report_id:
            report = self.get_report(report_id)
        else:
            if not self.current_report:
                raise ValueError("No active report")
            report = self.current_report
        
        citations = []
        
        # Group by source type
        sources_by_type = defaultdict(list)
        for attribution in report.source_attributions:
            sources_by_type[attribution.source_type].append(attribution)
        
        for source_type, sources in sources_by_type.items():
            citations.append(f"\n{source_type.upper()} Sources:")
            for i, source in enumerate(sources, 1):
                citations.append(
                    f"  [{i}] {source.source_identifier} "
                    f"(relevance: {source.relevance_score:.2f})"
                )
        
        return "\n".join(citations)
    
    def generate_visual_report(self, report_id: str = None) -> str:
        """Generate a visual text-based report"""
        if report_id:
            report = self.get_report(report_id)
        else:
            if not self.current_report:
                raise ValueError("No active report")
            report = self.current_report
        
        lines = [
            f"=== Tool Attribution Report: {report.report_id} ===",
            f"Query: {report.query}",
            f"Total Execution Time: {report.total_execution_time:.2f}s",
            "",
            "Answer Composition:",
        ]
        
        # Show composition as a bar chart
        for tool, percentage in sorted(
            report.answer_composition.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            lines.append(f"  {tool:20s} {bar} {percentage:.1f}%")
        
        lines.extend([
            "",
            "Tool Usage Summary:",
            f"  Total Tools Invoked: {report.tool_usage_summary.get('total_tools_invoked', 0)}",
            f"  Unique Tools Used: {report.tool_usage_summary.get('unique_tools_used', 0)}",
        ])
        
        if report.tool_usage_summary.get('most_used_tool'):
            lines.append(f"  Most Used Tool: {report.tool_usage_summary['most_used_tool']}")
        
        lines.extend([
            "",
            "Tools Used (in order):",
        ])
        
        for i, contribution in enumerate(report.tools_used, 1):
            status = "✓" if not contribution.error_occurred else "✗"
            lines.append(
                f"  {i}. {status} {contribution.tool_name} "
                f"({contribution.execution_duration:.2f}s, "
                f"contribution: {contribution.contribution_score:.2f})"
            )
        
        if report.source_attributions:
            lines.extend([
                "",
                "Source Attributions:",
            ])
            
            for i, source in enumerate(report.source_attributions, 1):
                lines.append(
                    f"  {i}. [{source.source_type}] {source.source_identifier} "
                    f"(relevance: {source.relevance_score:.2f})"
                )
        
        lines.extend([
            "",
            "Final Answer:",
            f"  {report.final_answer[:200]}{'...' if len(report.final_answer) > 200 else ''}"
        ])
        
        return "\n".join(lines)
    
    def analyze_tool_efficiency(self, report_ids: List[str]) -> Dict[str, Any]:
        """Analyze tool efficiency across multiple reports"""
        all_contributions = []
        
        for report_id in report_ids:
            try:
                report = self.get_report(report_id)
                all_contributions.extend(report.tools_used)
            except Exception as e:
                logger.warning(f"Could not load report {report_id}: {e}")
                continue
        
        if not all_contributions:
            return {}
        
        # Aggregate statistics
        tool_stats = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "contributions": [],
            "errors": 0,
            "successes": 0
        })
        
        for contribution in all_contributions:
            stats = tool_stats[contribution.tool_name]
            stats["count"] += 1
            stats["total_time"] += contribution.execution_duration
            stats["contributions"].append(contribution.contribution_score)
            
            if contribution.error_occurred:
                stats["errors"] += 1
            else:
                stats["successes"] += 1
        
        # Calculate derived metrics
        analysis = {}
        for tool_name, stats in tool_stats.items():
            analysis[tool_name] = {
                "total_invocations": stats["count"],
                "average_execution_time": stats["total_time"] / stats["count"],
                "average_contribution_score": sum(stats["contributions"]) / len(stats["contributions"]),
                "success_rate": (stats["successes"] / stats["count"]) * 100,
                "error_rate": (stats["errors"] / stats["count"]) * 100,
                "efficiency_score": (
                    (stats["successes"] / stats["count"]) * 
                    (sum(stats["contributions"]) / len(stats["contributions"])) * 
                    (1 / (stats["total_time"] / stats["count"]))
                )
            }
        
        return {
            "tool_statistics": analysis,
            "total_reports_analyzed": len(report_ids),
            "total_tool_invocations": len(all_contributions),
            "most_efficient_tool": max(
                analysis.items(),
                key=lambda x: x[1]["efficiency_score"]
            )[0] if analysis else None,
            "most_reliable_tool": max(
                analysis.items(),
                key=lambda x: x[1]["success_rate"]
            )[0] if analysis else None
        }
    
    def export_for_visualization(self, report_id: str) -> Dict[str, Any]:
        """Export data in a format suitable for visualization tools"""
        report = self.get_report(report_id)
        
        return {
            "report_id": report.report_id,
            "query": report.query,
            "nodes": [
                {
                    "id": f"tool_{i}",
                    "label": contrib.tool_name,
                    "type": contrib.tool_type,
                    "contribution": contrib.contribution_score,
                    "execution_time": contrib.execution_duration,
                    "error": contrib.error_occurred
                }
                for i, contrib in enumerate(report.tools_used)
            ],
            "edges": [
                {
                    "from": f"tool_{i}",
                    "to": f"tool_{i+1}",
                    "weight": 1
                }
                for i in range(len(report.tools_used) - 1)
            ],
            "composition": report.answer_composition,
            "metadata": {
                "timestamp": report.timestamp,
                "total_time": report.total_execution_time,
                "total_tools": len(report.tools_used)
            }
        }