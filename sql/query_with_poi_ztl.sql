-- Complete query with POI and ZTL features
WITH 
{poi_cte},
{ztl_cte}

SELECT
    {select_clause},
    {poi_selects},
    ZTL_INFO.InZTL as InZTL
FROM
    Atti A
    INNER JOIN AttiImmobili AI ON AI.IdAtto = A.Id
    INNER JOIN ParticelleCatastali PC ON AI.IdParticellaCatastale = PC.Id
    INNER JOIN IstatSezioniCensuarie2021 ISC ON PC.IdSezioneCensuaria = ISC.Id
    INNER JOIN IstatIndicatori2021 II ON II.IdIstatZonaCensuaria = ISC.Id
    INNER JOIN ParticelleCatastali_OmiZone PC_OZ ON PC_OZ.IdParticella = PC.Id
    INNER JOIN OmiZone OZ ON PC_OZ.IdZona = OZ.Id
    INNER JOIN OmiValori OVN ON OZ.Id = OVN.IdZona
        AND OVN.Stato = 'Normale'
        AND OVN.IdTipologiaEdilizia = CASE WHEN AI.IdTipologiaEdilizia = 8 THEN 2 ELSE AI.IdTipologiaEdilizia END
        AND A.Semestre = OZ.IdAnnoSemestre
    LEFT JOIN OmiValori OVO ON OZ.Id = OVO.IdZona
        AND OVO.Stato = 'Ottimo'
        AND OVO.IdTipologiaEdilizia = CASE WHEN AI.IdTipologiaEdilizia = 8 THEN 2 ELSE AI.IdTipologiaEdilizia END
        AND A.Semestre = OZ.IdAnnoSemestre
    LEFT JOIN OmiValori OVS ON OZ.Id = OVS.IdZona
        AND OVS.Stato = 'Scadente'
        AND OVS.IdTipologiaEdilizia = CASE WHEN AI.IdTipologiaEdilizia = 8 THEN 2 ELSE AI.IdTipologiaEdilizia END
        AND A.Semestre = OZ.IdAnnoSemestre
    -- CEN-ED View joins
    LEFT JOIN attiimmobili_cened1 C1 ON AI.Id = C1.IdAttoImmobile
    LEFT JOIN attiimmobili_cened2 C2 ON AI.Id = C2.IdAttoImmobile
    LEFT JOIN ZTL_CHECK ZTL_INFO ON PC.Id = ZTL_INFO.IdParticella
    {poi_joins}
WHERE 
    A.TotaleFabbricati = A.TotaleImmobili
    AND A.Id NOT IN (
        SELECT IdAtto
        FROM AttiImmobili
        WHERE Superficie IS NULL
    )
ORDER BY A.Id
