-- CTE for POI (Points of Interest) counts
-- Counts POI by category within the isodistance polygon of each parcel
POI_COUNTS AS (
    SELECT 
        PC_MAIN.Id as IdParticella,
        PDIT.Id as TipologiaPOI,
        PDIT.Denominazione as DenominazionePOI,
        COUNT(PDI.Id) as ConteggioPOI
    FROM 
        ParticelleCatastali PC_MAIN
        CROSS JOIN PuntiDiInteresseTipologie PDIT
        LEFT JOIN (
            PuntiDiInteresse PDI 
            INNER JOIN PuntiDiInteresse_Tipologie PDI_T ON PDI.Id = PDI_T.IdPuntoDiInteresse
        ) ON PDI_T.IdTipologia = PDIT.Id 
            AND PC_MAIN.Isodistanza.STContains(PDI.Posizione) = 1
    GROUP BY PC_MAIN.Id, PDIT.Id, PDIT.Denominazione
)
