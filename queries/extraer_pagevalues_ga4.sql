-- Datos históricos de Google Analytics 4
-- Periodo: noviembre de 2020

WITH eventos AS (
    SELECT
        user_pseudo_id,

        (SELECT MAX(value.int_value)
         FROM UNNEST(event_params)
         WHERE key = 'ga_session_id') AS id_sesion,

        event_timestamp,
        event_name,

        (SELECT MAX(value.string_value)
         FROM UNNEST(event_params)
         WHERE key = 'page_location') AS pagina,

        COALESCE(
            ecommerce.purchase_revenue_in_usd, 0
        ) AS ingresos

    FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`

    WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20201130'
      AND event_name IN ('page_view', 'purchase')
),

sesiones AS (
    SELECT
        user_pseudo_id,
        id_sesion,

        MIN(
            IF(event_name = 'purchase', event_timestamp, NULL)
        ) AS primera_compra,

        SUM(
            IF(event_name = 'purchase', ingresos, 0)
        ) AS ingresos_sesion

    FROM eventos
    WHERE user_pseudo_id IS NOT NULL
      AND user_pseudo_id NOT IN ('', '<Other>')
      AND id_sesion IS NOT NULL
    GROUP BY user_pseudo_id, id_sesion
),

paginas_por_sesion AS (
    SELECT DISTINCT
        REGEXP_REPLACE(e.pagina, r'[?#].*$', '') AS pagina,
        e.user_pseudo_id,
        e.id_sesion,
        s.ingresos_sesion,
        s.primera_compra IS NOT NULL AS compra

    FROM eventos AS e

    JOIN sesiones AS s
        USING (user_pseudo_id, id_sesion)

    WHERE e.event_name = 'page_view'
      AND e.pagina LIKE 'https://%'

      -- Excluir visitas posteriores a la primera compra
      AND (
          s.primera_compra IS NULL
          OR e.event_timestamp < s.primera_compra
      )
)

SELECT
    pagina,
    COUNT(*) AS sesiones_historicas,
    COUNTIF(compra) AS sesiones_compradoras,
    SUM(ingresos_sesion) AS ingresos_asociados

FROM paginas_por_sesion

GROUP BY pagina
ORDER BY sesiones_historicas DESC;