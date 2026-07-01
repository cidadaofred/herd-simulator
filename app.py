from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytical.dataset_package import DatasetPackage
from src.services.experiment_service import (
    ContinuationRequest,
    ExperimentService,
    SpecialistSimulationRequest,
)


st.set_page_config(
    page_title="Farm Monitoring AI",
    page_icon="🐄",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1280px;}
      [data-testid="stMetric"] {border: 1px solid #dce8df; padding: .8rem; border-radius: .7rem;}
      .small-note {color:#617066; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

service = ExperimentService()

if "default_campaign_id" not in st.session_state:
    st.session_state["default_campaign_id"] = f"exp_{datetime.now():%Y%m%d_%H%M%S}"
if "default_continuation_id" not in st.session_state:
    st.session_state["default_continuation_id"] = (
        f"continuacao_{datetime.now():%Y%m%d_%H%M%S}"
    )


def read_json(path):
    return DatasetPackage.read_json(Path(path))


def download_zip(directory, name, label, key):
    archive = service.create_zip(directory, name)
    with open(archive, "rb") as file:
        st.download_button(
            label,
            data=file.read(),
            file_name=archive.name,
            mime="application/zip",
            key=key,
        )


st.title("Farm Monitoring AI")
st.caption(
    "Digital Twin pecuário simplificado para geração controlada de datasets "
    "e avaliação independente de agentes analíticos."
)

with st.sidebar:
    st.subheader("Separação experimental")
    st.markdown(
        "O agente analítico recebe somente as observações. O ground truth é "
        "aberto posteriormente pelo avaliador."
    )
    st.info("MVP acadêmico: execuções locais, síncronas e isoladas por campanha.")

studio_tab, analysis_tab, management_tab, analysis_management_tab, method_tab = st.tabs(
    [
        "Modo Especialista",
        "Laboratório Analítico",
        "Gerenciar datasets",
        "Gerenciar análises",
        "Método",
    ]
)

with studio_tab:
    st.subheader("Construção do cenário")
    st.write(
        "Defina um experimento reproduzível. A composição nutricional e o lote-base "
        "serão ajustados automaticamente ao tamanho do rebanho."
    )

    with st.form("specialist_simulation"):
        first, second, third = st.columns(3)
        with first:
            campaign_id = st.text_input(
                "Identificador", value=st.session_state["default_campaign_id"]
            )
            num_days = st.slider("Dias simulados", 1, 15, 1)
            num_cattle = st.number_input(
                "Quantidade de bovinos", min_value=4, max_value=100, value=40
            )
        with second:
            seed = st.number_input("Semente aleatória", value=42, step=1)
            start_date = st.date_input("Data inicial", value=date.today())
            temperature_offset = st.slider(
                "Ajuste de temperatura (°C)", -15, 15, 0
            )
        with third:
            event_labels = list(service.EVENT_LABELS.values())
            selected_labels = st.multiselect(
                "Ocorrências controladas",
                event_labels,
                help="Só entram eventos agendados dentro da duração escolhida.",
            )
            st.markdown(
                '<p class="small-note">Datasets sem eventos também são úteis '
                "para medir falsos positivos.</p>",
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "Gerar simulação e snapshot", type="primary", width="stretch"
        )

    if submitted:
        reverse_labels = {label: key for key, label in service.EVENT_LABELS.items()}
        request = SpecialistSimulationRequest(
            campaign_id=campaign_id,
            num_days=int(num_days),
            num_cattle=int(num_cattle),
            seed=int(seed),
            start_date=start_date.isoformat(),
            temperature_offset_c=int(temperature_offset),
            event_types=tuple(reverse_labels[label] for label in selected_labels),
        )
        try:
            with st.status("Executando o experimento...", expanded=True) as status:
                st.write("Validando parâmetros e composição do rebanho")
                preview = service.config_preview(request)
                st.write("Gerando frames e metadados")
                result = service.run_simulation(request)
                status.update(label="Experimento concluído", state="complete")
            st.session_state["simulation_result"] = result
            st.session_state["simulation_preview"] = preview
        except Exception as exc:
            st.error(f"Não foi possível executar: {exc}")

    result = st.session_state.get("simulation_result")
    if result:
        st.success(f"Snapshot `{result.snapshot_id}` criado com sucesso.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames", result.frame_count)
        c2.metric("Intervalo", f"{result.frame_start}–{result.frame_end}")
        c3.metric("Dataset", result.snapshot_id)
        left, right = st.columns([1.4, 1])
        with left:
            st.image(str(result.preview_image), caption="Último frame processado")
        with right:
            with st.expander("Plano efetivamente executado"):
                st.json(st.session_state.get("simulation_preview", {}))
            download_zip(
                result.snapshot_root,
                f"simulation_{result.snapshot_id}",
                "Baixar simulation.zip",
                "download_simulation",
            )

    st.divider()
    st.subheader("Continuar um experimento")
    st.write(
        "Crie uma ramificação a partir do último checkpoint. O dataset original "
        "permanece intacto e o novo reúne todos os frames anteriores e futuros."
    )
    continuable = service.list_continuable_snapshots()
    if not continuable:
        st.info("Nenhum dataset com checkpoint final está disponível.")
    else:
        continuation_by_id = {
            item["dataset_id"]: item for item in continuable
        }
        base_dataset_id = st.selectbox(
            "Dataset de origem",
            list(continuation_by_id),
            key="continuation_base_dataset",
        )
        base_info = continuation_by_id[base_dataset_id]
        st.caption(
            f"Dia concluído: {base_info['completed_day']} · "
            f"último frame: {base_info['frame_end']} · "
            f"animais ativos: {base_info['active_animals']}"
        )
        with st.form("continue_experiment"):
            first, second, third = st.columns(3)
            with first:
                new_dataset_id = st.text_input(
                    "Novo identificador",
                    value=st.session_state["default_continuation_id"],
                )
                additional_days = st.slider("Dias adicionais", 1, 15, 3)
            with second:
                target_animals = st.number_input(
                    "Animais ativos após o ajuste",
                    min_value=1,
                    max_value=200,
                    value=int(base_info["active_animals"]),
                )
                continuation_seed = st.number_input(
                    "Seed da continuação", value=84, step=1
                )
            with third:
                continuation_event_labels = st.multiselect(
                    "Tipos de eventos aleatórios",
                    list(service.EVENT_LABELS.values()),
                )
                random_event_count = st.slider(
                    "Quantidade de eventos", 0, 10, 0
                )
            continue_submitted = st.form_submit_button(
                "Criar dataset composto", type="primary", width="stretch"
            )

        if continue_submitted:
            reverse_labels = {
                label: key for key, label in service.EVENT_LABELS.items()
            }
            continuation_request = ContinuationRequest(
                base_dataset_id=base_dataset_id,
                new_dataset_id=new_dataset_id,
                additional_days=int(additional_days),
                target_active_animals=int(target_animals),
                seed=int(continuation_seed),
                event_types=tuple(
                    reverse_labels[label]
                    for label in continuation_event_labels
                ),
                random_event_count=int(random_event_count),
            )
            try:
                with st.status(
                    "Continuando o experimento...", expanded=True
                ) as status:
                    st.write("Restaurando o último checkpoint")
                    st.write("Executando a ramificação e compondo os datasets")
                    continuation_result = service.continue_experiment(
                        continuation_request
                    )
                    status.update(
                        label="Dataset composto concluído", state="complete"
                    )
                st.session_state["continuation_result"] = continuation_result
            except Exception as exc:
                st.error(f"Não foi possível continuar: {exc}")

    continuation_result = st.session_state.get("continuation_result")
    if continuation_result:
        st.success(
            f"Dataset `{continuation_result.dataset_id}` criado a partir de "
            f"`{continuation_result.parent_dataset_id}`."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último frame original", continuation_result.parent_frame_end)
        c2.metric(
            "Frames acrescentados",
            continuation_result.extension_frame_end
            - continuation_result.extension_frame_start
            + 1,
        )
        c3.metric("Total de frames", continuation_result.total_frame_count)
        c4.metric(
            "Inventário",
            f"{continuation_result.active_animals_before} → "
            f"{continuation_result.active_animals_requested}",
        )
        left, right = st.columns([1.4, 1])
        with left:
            st.image(
                str(continuation_result.preview_image),
                caption="Último frame da continuação",
            )
        with right:
            st.code(
                f"{continuation_result.parent_dataset_id}\n"
                f"  + {continuation_result.branch_campaign_id}\n"
                f"  = {continuation_result.dataset_id}",
                language="text",
            )
            download_zip(
                continuation_result.snapshot_root,
                f"simulation_{continuation_result.dataset_id}",
                "Baixar dataset composto",
                "download_composite_simulation",
            )

with analysis_tab:
    st.subheader("Analisar um dataset fechado")
    snapshots = service.list_snapshots()
    if not snapshots:
        st.warning("Nenhum snapshot v2 disponível. Gere uma simulação primeiro.")
    else:
        selected_dataset = st.selectbox("Dataset", snapshots)
        st.caption(
            "A inferência é executada sobre o manifesto observável; a comparação "
            "com o ground truth ocorre somente na etapa de avaliação."
        )
        if st.button("Executar análise e avaliação", type="primary"):
            try:
                with st.status("Analisando o dataset...", expanded=True) as status:
                    st.write("Inferência temporal e consolidação dos alertas")
                    analysis_result = service.run_analysis(selected_dataset)
                    st.write("Avaliação independente contra o ground truth")
                    status.update(label="Análise concluída", state="complete")
                st.session_state["analysis_result"] = analysis_result
            except Exception as exc:
                st.error(f"Falha na análise: {exc}")

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result:
        evaluation = read_json(analysis_result.evaluation_path)
        alerts = read_json(analysis_result.alerts_path)
        summary = read_json(analysis_result.summary_path)
        population_path = analysis_result.output_dir / "population_history.json"
        population_history = (
            read_json(population_path) if population_path.is_file() else []
        )
        st.success(
            f"Análise `{analysis_result.run_id}` concluída para "
            f"`{analysis_result.dataset_id}`."
        )
        metrics = st.columns(6)
        metrics[0].metric("TP", evaluation["true_positives"])
        metrics[1].metric("FP", evaluation["false_positives"])
        metrics[2].metric("FN", evaluation["false_negatives"])
        metrics[3].metric("Precision", f"{evaluation['precision']:.2f}")
        metrics[4].metric("Recall", f"{evaluation['recall']:.2f}")
        metrics[5].metric("F1", f"{evaluation['f1_score']:.2f}")
        population_metrics = st.columns(2)
        population_metrics[0].metric(
            "População visual estimada",
            summary.get("final_observed_population_estimate", "—"),
        )
        population_metrics[1].metric(
            "Mudanças persistentes observadas",
            summary.get("observed_population_change_count", 0),
        )
        with st.expander("Evolução da população observada"):
            st.caption(
                "Esta referência deriva apenas dos tracks persistentes. Ela não "
                "classifica a causa como venda, entrada, desaparecimento ou morte."
            )
            if population_history:
                st.dataframe(
                    pd.DataFrame(population_history),
                    width="stretch",
                    hide_index=True,
                )
        st.markdown("#### Alertas e ocorrências")
        alert_rows = service.evaluation_alert_rows(alerts, evaluation)
        if alert_rows:
            alert_table = pd.DataFrame(alert_rows)

            def highlight_status(row):
                if row["status"] == "Não detectado":
                    style = "background-color: #7f1d1d; color: #fff; font-weight: 600;"
                elif row["status"] == "Falso positivo":
                    style = "background-color: #713f12; color: #fff;"
                else:
                    style = ""
                return [style] * len(row)

            styled_table = alert_table.style.apply(highlight_status, axis=1)
            st.dataframe(
                styled_table,
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "🔴 ocorrência não detectada · 🟡 alerta falso positivo · "
                "linhas sem cor: ocorrências detectadas"
            )
        else:
            st.info("Não há alertas nem ocorrências avaliáveis neste dataset.")
        download_zip(
            analysis_result.output_dir,
            f"analysis_{analysis_result.dataset_id}_{analysis_result.run_id}",
            "Baixar analysis.zip",
            "download_analysis",
        )

with management_tab:
    st.subheader("Gerenciar datasets no servidor")
    st.warning(
        "A exclusão é permanente. Ela remove o snapshot, análises, ZIPs, "
        "segmentos e campanhas exclusivos, além de continuações descendentes."
    )
    deletion_message = st.session_state.pop("dataset_deletion_message", None)
    if deletion_message:
        st.success(deletion_message)

    managed_datasets = service.list_snapshots()
    if not managed_datasets:
        st.info("Não existem datasets armazenados no servidor.")
    else:
        managed_dataset = st.selectbox(
            "Dataset para excluir",
            managed_datasets,
            key="managed_dataset",
        )
        deletion_preview = service.dataset_deletion_preview(managed_dataset)
        preview_columns = st.columns(5)
        preview_columns[0].metric("Datasets", len(deletion_preview.dataset_ids))
        preview_columns[1].metric(
            "Análises", deletion_preview.analysis_run_count
        )
        preview_columns[2].metric("Segmentos", deletion_preview.segment_count)
        preview_columns[3].metric("Campanhas", deletion_preview.campaign_count)
        preview_columns[4].metric(
            "Espaço estimado",
            f"{deletion_preview.total_bytes / (1024 * 1024):.1f} MB",
        )
        if len(deletion_preview.dataset_ids) > 1:
            descendant_ids = [
                item
                for item in deletion_preview.dataset_ids
                if item != managed_dataset
            ]
            st.error(
                "Este dataset possui continuações dependentes. Também serão "
                "excluídos: "
                + ", ".join(descendant_ids)
            )
        with st.expander("Conteúdo abrangido pela exclusão"):
            st.write("Datasets: " + ", ".join(deletion_preview.dataset_ids))
            st.write(f"ZIPs exportados: {deletion_preview.export_count}")
            st.write(
                "Campanhas compartilhadas com datasets preservados não serão removidas."
            )

        with st.form("delete_dataset_form"):
            confirmation = st.text_input(
                f"Digite `{managed_dataset}` para confirmar",
                key="dataset_delete_confirmation",
            )
            understood = st.checkbox(
                "Entendo que esta operação não pode ser desfeita."
            )
            delete_submitted = st.form_submit_button(
                "Excluir definitivamente",
                type="primary",
            )
        if delete_submitted:
            if not understood:
                st.error("Marque a confirmação de que a operação é permanente.")
            elif confirmation.strip() != managed_dataset:
                st.error("O texto de confirmação deve ser exatamente o ID selecionado.")
            else:
                try:
                    result = service.delete_dataset(managed_dataset, confirmation)
                    current_analysis = st.session_state.get("analysis_result")
                    if (
                        current_analysis
                        and current_analysis.dataset_id in result.deleted_dataset_ids
                    ):
                        st.session_state.pop("analysis_result", None)
                    current_continuation = st.session_state.get("continuation_result")
                    if (
                        current_continuation
                        and current_continuation.dataset_id
                        in result.deleted_dataset_ids
                    ):
                        st.session_state.pop("continuation_result", None)
                    st.session_state["dataset_deletion_message"] = (
                        f"Exclusão concluída: {len(result.deleted_dataset_ids)} "
                        f"dataset(s) e {result.reclaimed_bytes / (1024 * 1024):.1f} MB removidos."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível excluir: {exc}")

with analysis_management_tab:
    st.subheader("Gerenciar análises no servidor")
    st.info(
        "A exclusão remove somente a execução analítica selecionada e seu ZIP. "
        "O dataset de simulação permanece disponível para novas análises."
    )
    analysis_deletion_message = st.session_state.pop(
        "analysis_deletion_message", None
    )
    if analysis_deletion_message:
        st.success(analysis_deletion_message)

    stored_analyses = service.list_analysis_runs()
    if not stored_analyses:
        st.info("Não existem análises armazenadas no servidor.")
    else:
        analysis_datasets = sorted(
            {item["dataset_id"] for item in stored_analyses}
        )
        managed_analysis_dataset = st.selectbox(
            "Dataset analisado",
            analysis_datasets,
            key="managed_analysis_dataset",
        )
        dataset_runs = [
            item
            for item in stored_analyses
            if item["dataset_id"] == managed_analysis_dataset
        ]
        run_by_id = {item["run_id"]: item for item in dataset_runs}
        managed_run_id = st.selectbox(
            "Execução analítica",
            list(run_by_id),
            key="managed_analysis_run",
        )
        managed_run = run_by_id[managed_run_id]
        run_preview = service.analysis_deletion_preview(
            managed_analysis_dataset,
            managed_run_id,
        )
        run_columns = st.columns(5)
        run_columns[0].metric(
            "Precision",
            "—" if managed_run["precision"] is None else managed_run["precision"],
        )
        run_columns[1].metric(
            "Recall",
            "—" if managed_run["recall"] is None else managed_run["recall"],
        )
        run_columns[2].metric(
            "F1",
            "—" if managed_run["f1_score"] is None else managed_run["f1_score"],
        )
        run_columns[3].metric(
            "Alertas",
            "—" if managed_run["alert_count"] is None else managed_run["alert_count"],
        )
        run_columns[4].metric(
            "Espaço estimado",
            f"{run_preview.total_bytes / (1024 * 1024):.1f} MB",
        )
        st.caption(
            f"Finalizada em: {managed_run['finished_at'] or 'não informado'} · "
            f"ZIPs associados: {run_preview.export_count}"
        )
        st.warning(
            f"Será excluída apenas a análise `{managed_run_id}` do dataset "
            f"`{managed_analysis_dataset}`."
        )
        with st.form("delete_analysis_form"):
            analysis_confirmation = st.text_input(
                f"Digite `{managed_run_id}` para confirmar",
                key="analysis_delete_confirmation",
            )
            analysis_understood = st.checkbox(
                "Entendo que esta análise será removida permanentemente."
            )
            analysis_delete_submitted = st.form_submit_button(
                "Excluir análise definitivamente",
                type="primary",
            )
        if analysis_delete_submitted:
            if not analysis_understood:
                st.error("Marque a confirmação de exclusão permanente.")
            elif analysis_confirmation.strip() != managed_run_id:
                st.error("O texto deve ser exatamente o run_id selecionado.")
            else:
                try:
                    result = service.delete_analysis(
                        managed_analysis_dataset,
                        managed_run_id,
                        analysis_confirmation,
                    )
                    current_analysis = st.session_state.get("analysis_result")
                    if (
                        current_analysis
                        and current_analysis.dataset_id == result.dataset_id
                        and current_analysis.run_id == result.run_id
                    ):
                        st.session_state.pop("analysis_result", None)
                    st.session_state["analysis_deletion_message"] = (
                        f"Análise `{result.run_id}` excluída; "
                        f"{result.reclaimed_bytes / (1024 * 1024):.1f} MB liberados."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Não foi possível excluir a análise: {exc}")

with method_tab:
    st.subheader("Como o experimento preserva independência")
    st.markdown(
        """
        1. O simulador registra o estado completo e gera as evidências observáveis.
        2. O snapshot cria `observable/` e `private/ground_truth` separados.
        3. O agente analítico recebe apenas bounding boxes, tracking, clima e imagens.
        4. Alertas são formados por evidências temporais, sem consultar ocorrências reais.
        5. O avaliador compara os alertas prontos com o ground truth e calcula as métricas.

        A identidade de tracking é uma hipótese explícita do pipeline sintético atual.
        Em uma evolução futura, ela poderá ser substituída por um rastreador real.
        """
    )
