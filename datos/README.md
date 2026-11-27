# datos

| Campo | Valor |
|---|---|
| Archivo | [`electricidad_residencial_eeuu.csv`](electricidad_residencial_eeuu.csv) |
| Qué mide | Ventas mensuales de electricidad a clientes del sector residencial de Estados Unidos, expresadas en miles de MWh. |
| Fuente | U.S. Energy Information Administration (EIA), [API de datos abiertos](https://www.eia.gov/opendata/documentation.php). Consulta de ventas minoristas de electricidad con los filtros `stateid=US`, `sectorid=RES` y frecuencia mensual. |
| Licencia | Datos de dominio público. La EIA permite su reutilización y solicita atribución. [Política de reutilización de la EIA](https://www.eia.gov/about/copyrights_reuse.php). |
| Frecuencia | Mensual. |
| Periodo | Enero de 2016 a diciembre de 2025, con 120 observaciones consecutivas. |
| Columna de fecha | `fecha`. Primer día de cada mes, en formato `AAAA-MM-DD`. |
| Columna que se pronostica | `ventas_miles_mwh`. Total mensual de ventas residenciales, en miles de MWh. |

La serie se conserva tal como viene en el CSV: 120 meses consecutivos, sin
vacíos y sin imputaciones ni recortes. La revisión del Miniproyecto 1 señaló
observaciones inusuales por mes calendario, sin evidencia de errores de
registro, así que se mantienen los valores originales.
