"""
SQL template loader and builder utilities.
Centralizes SQL query management for better maintainability.
"""

from pathlib import Path
from typing import Dict, List, Optional


class SQLTemplateLoader:
    """Loads and manages SQL query templates."""
    
    def __init__(self, templates_dir: str = "sql"):
        """
        Initialize the SQL template loader.
        
        Args:
            templates_dir: Directory containing SQL template files
        """
        self.templates_dir = Path(templates_dir)
        if not self.templates_dir.is_absolute():
            # Resolve relative to project root
            self.templates_dir = Path(__file__).parent.parent.parent / templates_dir
    
    def load_template(self, template_name: str) -> str:
        """
        Load a SQL template file.
        
        Args:
            template_name: Name of the template file (e.g., 'base_query.sql')
        
        Returns:
            Template content as string
        """
        template_path = self.templates_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"SQL template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def build_poi_selects(self, poi_categories: List[str]) -> str:
        """
        Build SELECT clause for POI count columns.
        
        Args:
            poi_categories: List of POI category IDs
        
        Returns:
            Comma-separated SELECT statements for POI counts
        """
        if not poi_categories:
            return "0 as no_poi"
        
        poi_selects = []
        for category in poi_categories:
            safe_category = (
                str(category).replace("-", "_")
                .replace(" ", "_")
                .replace(".", "_")
            )
            alias = f"POI_{safe_category}"
            poi_selects.append(
                f"COALESCE({alias}.ConteggioPOI, 0) AS POI_{safe_category}_count"
            )
        
        return ",\n        ".join(poi_selects)
    
    def build_poi_joins(self, poi_categories: List[str]) -> str:
        """
        Build LEFT JOIN clauses for POI counts.
        
        Args:
            poi_categories: List of POI category IDs
        
        Returns:
            SQL LEFT JOIN statements for POI counts
        """
        if not poi_categories:
            return ""
        
        poi_joins = []
        for category in poi_categories:
            safe_category = (
                str(category).replace("-", "_")
                .replace(" ", "_")
                .replace(".", "_")
            )
            alias = f"POI_{safe_category}"
            poi_joins.append(
                f"""
            LEFT JOIN POI_COUNTS {alias} ON PC.Id = {alias}.IdParticella 
                AND {alias}.TipologiaPOI = '{category}'"""
            )
        
        return "".join(poi_joins)
    
    def build_query_with_poi_ztl(
        self,
        select_clause: str,
        poi_categories: List[str],
        include_poi: bool = True,
        include_ztl: bool = True
    ) -> str:
        """
        Build complete query with optional POI and ZTL features.
        
        Args:
            select_clause: Main SELECT clause (column selections)
            poi_categories: List of POI category IDs
            include_poi: Whether to include POI features
            include_ztl: Whether to include ZTL features
        
        Returns:
            Complete SQL query string
        """
        ctes = []
        
        if include_poi:
            poi_cte = self.load_template('poi_counts_cte.sql')
            ctes.append(poi_cte)
        
        if include_ztl:
            ztl_cte = self.load_template('ztl_check_cte.sql')
            ctes.append(ztl_cte)
        
        if not ctes:
            # No POI/ZTL, use base query
            base_template = self.load_template('base_query.sql')
            return base_template.format(select_clause=select_clause)
        
        # Build query with CTEs
        query_template = self.load_template('query_with_poi_ztl.sql')
        
        poi_selects = self.build_poi_selects(poi_categories if include_poi else [])
        poi_joins = self.build_poi_joins(poi_categories if include_poi else [])
        
        cte_clause = ",\n".join(ctes)
        
        return query_template.format(
            poi_cte=ctes[0] if include_poi else "",
            ztl_cte=ctes[1] if (include_ztl and len(ctes) > 1) else (ctes[0] if include_ztl else ""),
            select_clause=select_clause,
            poi_selects=poi_selects,
            poi_joins=poi_joins
        )
    
    def build_base_query(self, select_clause: str) -> str:
        """
        Build base query without POI/ZTL features.
        
        Args:
            select_clause: Main SELECT clause (column selections)
        
        Returns:
            Complete SQL query string
        """
        template = self.load_template('base_query.sql')
        return template.format(select_clause=select_clause)
