from .utils import (
    add_slurm_skeleton,
    check_dd_freq,
    cwl_file,
    cwl_dir,
    extract_obsid_from_ms,
    get_container_env_var,
    get_prefactor_freqs,
    ra_dec_to_iltj,
    setup_toil_slurm,
    verify_slurm_environment_toil,
)
import glob
import json
import os
import re
import sys
import shutil
import structlog
import subprocess
import tempfile
from astropy.table import Table
from time import gmtime, strftime
from cyclopts import App, Parameter, Token
from enum import Enum
from typing import List, Optional, Tuple, Annotated, Literal

class VLBIJSONConfig:
    """Class for generating JSON configuration files to be passed to the VLBI-cwl pipeline."""

    class OBS_TYPE(Enum):
        DELAY = "delay-calibration"
        DD_CALIBRATION = "dd-calibration"
        SPLIT_DIRECTIONS = "split-directions"
        POLARIZATION_IMAGING = "polarization-imaging"
        SETUP = "setup"
        CONCATENATE_FLAG = "concatenate-flag"
        PHASEUP_CONCAT = "phaseup-concat"

    def __init__(
        self,
        mspath: str,
        ms_suffix: str = ".MS",
        prefac_h5parm={"path": ""},
        ddf_solsdir: dict = {"path": ""},
        update_version_file: bool = False,
        outdir: str = os.getcwd(),
    ):
        if "VLBI_DATA_ROOT" not in os.environ:
            logger.critical(
                "VLBI_DATA_ROOT environment variable has not been set. This is needed for pipeline execution."
            )
            sys.exit(-1)

        self.configdict = {}
        self.outdir = outdir
        """
        filedir = os.path.join(mspath, f"*{ms_suffix}")
        logger.info(f"Searching {filedir}")
        files = sorted(glob.glob(filedir))
        if not files:
            logger.critical(f"No MS files found in {mspath} with suffix {ms_suffix}")
            sys.exit(-1)
        logger.info(f"Found {len(files)} files")
		"""
		if not is_ms(mspath):
            filedir = os.path.join(mspath, f"*{ms_suffix}")
            logger.info(f"Searching {filedir}")
            files = sorted(glob.glob(filedir))
            if files:
                logger.info(f"Found {len(files)} files")
            else:
                logger.critical(
                    f"No MS files found in {mspath} with suffix {ms_suffix}."
                )
            sys.exit(-1)
        else:
            logger.info("Found MS {mspath}")
            files = [mspath]
            
        if (not prefac_h5parm) or (
            not prefac_h5parm["path"].endswith("h5")
            and not prefac_h5parm["path"].endswith("h5parm")
        ):
            mslist = []
            for ms in files:
                x = json.loads(f'{{"class": "Directory", "path":"{ms}"}}')
                mslist.append(x)
            self.configdict["msin"] = mslist

        else:
            prefac_freqs = get_prefactor_freqs(
                solname=prefac_h5parm["path"], solset="target"
            )

            mslist = []
            for dd in files:
                if check_dd_freq(dd, prefac_freqs):
                    mslist.append(dd)

            final_mslist = []
            for ms in mslist:
                x = json.loads(f'{{"class": "Directory", "path":"{ms}"}}')
                final_mslist.append(x)
            self.configdict["msin"] = final_mslist
        if ddf_solsdir and ddf_solsdir["path"]:
            soldirs = glob.glob(os.path.join(ddf_solsdir["path"], "L*"))
            ddf_names_short = []
            for ms in soldirs:
                short_name = "_".join(os.path.basename(ms).split("_")[:2])
                ddf_names_short.append(short_name)

            idx_nosols = set()
            for idx, ms in enumerate(self.configdict["msin"]):
                short_name = "_".join(os.path.basename(ms["path"]).split("_")[:2])
                if short_name not in ddf_names_short:
                    idx_nosols.add(idx)
                    logger.info(
                        f"Removing {os.path.basename(ms['path'])} because no ddf-pipeline solutions"
                    )
            self.configdict["msin"] = [
                ms
                for i, ms in enumerate(self.configdict["msin"])
                if i not in idx_nosols
            ]

        try:
            self.obsid = extract_obsid_from_ms(self.configdict["msin"][0]["path"])
        except IndexError:
            self.obsid = "unknown"
        self.create_linc_versions_file(update_version_file)

    def add_entry(self, key: str, value: object):
        if "ATeam" in key:
            self.configdict["Ateam_skymodel"] = value
        else:
            self.configdict[key] = value

    def create_linc_versions_file(self, overwrite=False):
        if "LINC_DATA_ROOT" not in os.environ:
            raise ValueError(
                "WARNING: LINC_DATA_ROOT environment variable has not been set. Cannot generate $LINC_DATA_ROOT/.versions file."
            )
        linc_version = subprocess.check_output(
            f"cd {os.environ['LINC_DATA_ROOT']} && git describe --tags",
            shell=True,
            text=True,
        )
        pip_versions = subprocess.check_output(
            "pip freeze | sed 's/==/: /g'", shell=True
        )
        linc_version_file = os.path.join(os.environ["LINC_DATA_ROOT"], ".versions")

        if os.path.isfile(linc_version_file) and not overwrite:
            logger.info(f"Using existing {os.environ['LINC_DATA_ROOT']}/.versions")
        if not os.path.isfile(linc_version_file) or overwrite:
            with open(linc_version_file, "wb") as f:
                f.write(f"LINC: {linc_version}".encode("utf-8"))
                f.write(pip_versions)

    def save(self, fname: str):
        if not fname.endswith(".json"):
            fname += ".json"
        with open(fname, "w") as outfile:
            json.dump(self.configdict, outfile, indent=4)
        logger.info(f"Written configuration to {fname}")
        self.configfile = fname

    def setup_rundir(self, workdir):
        self.rundir = tempfile.mkdtemp(
            prefix=f"tmp.VLBI_{self.mode.value}.", dir=workdir
        )

    def deduce_pipeline_mode(self):
        if self.configfile is None:
            raise RuntimeError("No config file has been created. Save it first.")
        elif "delay" in self.configfile:
            self.mode = self.OBS_TYPE.DELAY
        elif "dd-calibration" in self.configfile:
            self.mode = self.OBS_TYPE.DD_CALIBRATION
        elif "split" in self.configfile:
            self.mode = self.OBS_TYPE.SPLIT_DIRECTIONS
        elif "polarization" in self.configfile:
            self.mode = self.OBS_TYPE.POLARIZATION_IMAGING
        elif "setup" in self.configfile:
            self.mode = self.OBS_TYPE.SETUP
        elif "concatenate-flag" in self.configfile:
            self.mode = self.OBS_TYPE.CONCATENATE_FLAG
        elif "phaseup-concat" in self.configfile:
            self.mode = self.OBS_TYPE.PHASEUP_CONCAT
        else:
            raise RuntimeError(
                "Failed to deduce workflow from config file. Is it named correctly?"
            )

    def move_results_from_rundir(self):
        date = strftime("%Y_%m_%d-%H_%M_%S", gmtime())
        try:
            logger.info("Tarring log directory to reduce files")
            tarjob = subprocess.check_output(
                [
                    "tar",
                    "cf",
                    os.path.join(self.rundir, f"logs_VLBI_{self.mode.value}.tar"),
                    os.path.join(self.rundir, f"logs_VLBI_{self.mode.value}"),
                ]
            )
            logger.info("Removing log directory")
            subprocess.check_output(
                ["rm", "-r", os.path.join(self.rundir, f"logs_VLBI_{self.mode.value}")]
            )

            tempdirs = glob.glob(os.path.join(self.rundir, "toilwf-*"))
            for td in tempdirs:
                subprocess.check_output(["rm", "-rf", td])
        except subprocess.CalledProcessError:
            logger.warning("Failed to tar logs.")

        try:
            logger.info("Removing leftover tmpdirs")
            tempdirs = glob.glob(os.path.join(self.rundir, "tmpdir*"))
            for td in tempdirs:
                subprocess.check_output(["rm", "-rf", td])

            tempdirs = glob.glob(os.path.join(self.rundir, "toilwf-*"))
            for td in tempdirs:
                subprocess.check_output(["rm", "-rf", td])
        except subprocess.CalledProcessError:
            logger.warning("Failed to remove leftover tmpdirs.")

        logger.info("Copying results")
        shutil.move(
            self.rundir,
            os.path.join(self.outdir, f"VLBI_{self.mode.value}_L{self.obsid}_{date}"),
        )

    def run_workflow(
        self,
        runner: str = "toil",
        scheduler: str = "slurm",
        workdir: str = os.getcwd(),
        slurm_params: dict = {},
        restart: bool = False,
        record_stats: bool = False,
    ):
        self.deduce_pipeline_mode()
        if not restart:
            self.restarting = False
            self.setup_rundir(workdir)
        else:
            self.restarting = True
            self.rundir = workdir
            logger.info(f"Attempting to restart existing workflow from {self.rundir}.")
        self.setup_apptainer_variables(self.rundir)
        logger.info(
            f"Running workflow with {runner} under {scheduler} in {self.rundir}"
        )

        if runner == "cwltool":
            cmd = (
                "cwltool "
                + "--parallel "
                + "--timestamps "
                + "--disable-pull "
                + "--singularity "
                + f"--tmpdir-prefix={os.environ['APPTAINERENV_TMPDIR']} "
                + f"--outdir={os.environ['APPTAINERENV_RESULTSDIR']} "
                + f"--log-dir={os.environ['APPTAINERENV_LOGSDIR']} "
            )
            cmd += f"{os.environ['VLBI_DATA_ROOT']}/workflows/{self.mode.value}.cwl "
            cmd += f"{self.configfile}"

            if scheduler == "slurm":
                wrapped_cmd = add_slurm_skeleton(
                    contents=cmd,
                    job_name=f"VLBI_{self.mode.value}",
                    **slurm_params,
                )
                with open("temp_jobscript.sh", "w") as f:
                    f.write(wrapped_cmd)
                logger.info("Written temporary jobscript to temp_jobscript.sh")
                out = subprocess.check_output(["sbatch", "temp_jobscript.sh"]).decode(
                    "utf-8"
                )
            elif scheduler == "singleMachine":
                logger.info(f"Running command:\n{cmd}")
                out = subprocess.check_output(cmd.split(" ")).decode("utf-8")
                print(out)
        elif runner == "toil":
            verify_slurm_environment_toil()
            dir_coordination, dir_slurmlogs = self.setup_toil_directories(self.rundir)
            is_ceph = "ceph" in subprocess.check_output(
                ["df", self.rundir]
            ).lower().decode("utf-8")
            setup_toil_slurm(slurm_params)
            cmd = ["toil-cwl-runner"]
            if scheduler == "slurm":
                cmd += ["--batchSystem", "slurm"]
                cmd += ["--slurmTime", slurm_params["time"]]
                cmd += ["--slurmPartition", slurm_params["queue"]]
            elif scheduler == "singleMachine":
                cmd += ["--batchSystem", "single_machine"]
            else:
                raise ValueError(f"Unsupported scheduler `{scheduler}` provided.")
            if self.restarting:
                cmd += ["--restart"]
            if "TOIL_SLURM_ARGS" in os.environ.keys():
                cmd += [f"--slurmArgs='{os.environ['TOIL_SLURM_ARGS']}'"]
            if record_stats:
                cmd += ["--stats"]
            cmd += ["--no-cwl-default-ram"]
            cmd += ["--defaultMemory", "8Gi"]
            cmd += ["--no-read-only"]
            cmd += ["--retryCount", "1"]
            cmd += ["--singularity"]
            cmd += ["--disableCaching"]
            cmd += ["--writeLogsFromAllJobs", "True"]
            cmd += [
                "--logFile",
                os.path.join(
                    self.rundir, f"full_log_{self.mode.value}_{self.obsid}.log"
                ),
            ]
            cmd += ["--writeLogs", get_container_env_var("LOGSDIR")]
            cmd += ["--outdir", get_container_env_var("RESULTSDIR")]
            cmd += ["--tmp-outdir-prefix", get_container_env_var("TMPDIR")]
            cmd += ["--jobStore", os.path.join(self.rundir, "jobstore")]
            cmd += ["--workDir", workdir]
            if is_ceph:
                logger.info("Detected CEPH file system, not setting coordinationDir.")
            else:
                cmd += ["--coordinationDir", dir_coordination]
            cmd += ["--tmpdir-prefix", get_container_env_var("TMPDIR")]
            cmd += ["--disableAutoDeployment", "True"]
            cmd += ["--bypass-file-store"]
            cmd += [
                "--batchLogsDir",
                os.path.join(get_container_env_var("LOGSDIR"), dir_slurmlogs),
            ]
            cmd += ["--no-compute-checksum"]
            cmd += ["--moveOutputs", "True"]
            cmd += [
                os.path.join(
                    os.environ["VLBI_DATA_ROOT"], "workflows", f"{self.mode.value}.cwl"
                )
            ]
            cmd += [self.configfile]
            try:
                out = subprocess.check_output(cmd)
                with open(f"log_VLBI_{self.mode.value}.txt", "wb") as f:
                    f.write(out)
                self.move_results_from_rundir()
            except subprocess.CalledProcessError as e:
                with open(f"log_VLBI_{self.mode.value}.txt", "wb") as f:
                    f.write(e.stdout)
                if e.stderr:
                    with open(f"log_VLBI_{self.mode.value}_err.txt", "wb") as f:
                        f.write(e.stderr)

    def setup_apptainer_variables(self, workdir):
        try:
            out = (
                subprocess.check_output(["singularity", "--version"])
                .decode("utf-8")
                .strip()
            )
        except subprocess.CalledProcessError:
            out = (
                subprocess.check_output(["apptainer", "--version"])
                .decode("utf-8")
                .strip()
            )
        if "apptainer" in out:
            os.environ["APPTAINERENV_VLBI_DATA_ROOT"] = os.environ["VLBI_DATA_ROOT"]
            os.environ["APPTAINERENV_LINC_DATA_ROOT"] = os.environ["LINC_DATA_ROOT"]
            os.environ["APPTAINERENV_RESULTSDIR"] = (
                f"{workdir}/results_VLBI_{self.mode.value}/"
            )
            os.environ["APPTAINERENV_LOGSDIR"] = (
                f"{workdir}/logs_VLBI_{self.mode.value}/"
            )
            os.environ["APPTAINERENV_TMPDIR"] = (
                f"{workdir}/tmpdir_VLBI_{self.mode.value}/"
            )
            os.environ["APPTAINERENV_PREPEND_PATH"] = (
                f"{os.environ['VLBI_DATA_ROOT']}/scripts:{os.environ['LINC_DATA_ROOT']}/scripts"
            )
            if not self.restarting:
                os.mkdir(os.environ["APPTAINERENV_LOGSDIR"])
                os.mkdir(os.environ["APPTAINERENV_TMPDIR"])
                os.mkdir(os.environ["APPTAINERENV_RESULTSDIR"])
            if "APPTAINER_BINDPATH" not in os.environ:
                os.environ["APPTAINER_BINDPATH"] = (
                    f"{os.environ['LINC_DATA_ROOT']}:/opt/lofar/LINC"
                    + f",{os.environ['LINC_DATA_ROOT']}:/opt/lofar/VLBI-cwl"  # VLBI-cwl is earlier in PATH, this is intentional.
                    + f",{os.path.dirname(workdir)}"
                )
            else:
                os.environ["APPTAINER_BINDPATH"] = (
                    f"{os.environ['LINC_DATA_ROOT']}:/opt/lofar/LINC"
                    + f",{os.environ['LINC_DATA_ROOT']}:/opt/lofar/VLBI-cwl"  # VLBI-cwl is earlier in PATH, this is intentional.
                    + f",{workdir}"
                    + f",{os.environ['APPTAINER_BINDPATH']}"
                )
        elif "singularity" in out:
            os.environ["SINGULARITYENV_VLBI_DATA_ROOT"] = os.environ["VLBI_DATA_ROOT"]
            os.environ["SINGULARITYENV_LINC_DATA_ROOT"] = os.environ["LINC_DATA_ROOT"]
            os.environ["SINGULARITYENV_RESULTSDIR"] = (
                f"{workdir}/results_VLBI_{self.mode.value}/"
            )
            os.environ["SINGULARITYENV_LOGSDIR"] = (
                f"{workdir}/logs_VLBI_{self.mode.value}/"
            )
            os.environ["SINGULARITYENV_TMPDIR"] = (
                f"{workdir}/tmpdir_VLBI_{self.mode.value}/"
            )
            os.environ["SINGULARITYENV_PREPEND_PATH"] = (
                f"{os.environ['VLBI_DATA_ROOT']}/scripts:{os.environ['LINC_DATA_ROOT']}/scripts"
            )
            if not self.restarting:
                os.mkdir(os.environ["SINGULARITYENV_LOGSDIR"])
                os.mkdir(os.environ["SINGULARITYENV_TMPDIR"])
                os.mkdir(os.environ["SINGULARITYENV_RESULTSDIR"])
            if "SINGULARITY_BINDPATH" not in os.environ:
                os.environ["SINGULARITY_BINDPATH"] = (
                    f"{os.environ['LINC_DATA_ROOT']}:/opt/lofar/LINC"
                    + f",{os.environ['LINC_DATA_ROOT']}:/opt/lofar/VLBI-cwl"  # VLBI-cwl is earlier in PATH, this is intentional.
                    + f",{os.path.dirname(workdir)}"
                )
            else:
                os.environ["SINGULARITY_BINDPATH"] = (
                    f"{os.environ['LINC_DATA_ROOT']}:/opt/lofar/LINC"
                    + f",{os.environ['LINC_DATA_ROOT']}:/opt/lofar/VLBI-cwl"  # VLBI-cwl is earlier in PATH, this is intentional.
                    + f",{workdir}"
                    + f",{os.environ['SINGULARITY_BINDPATH']}"
                )
        if "PYTHONPATH" in os.environ:
            os.environ["PYTHONPATH"] = (
                "$LINC_DATA_ROOT/scripts:" + os.environ["PYTHONPATH"]
            )
            os.environ["PYTHONPATH"] = (
                "$VLBI_DATA_ROOT/scripts:" + os.environ["PYTHONPATH"]
            )
        else:
            os.environ["PYTHONPATH"] = "$LINC_DATA_ROOT/scripts"
            os.environ["PYTHONPATH"] = "$VLBI_DATA_ROOT/scripts"
        os.environ["PATH"] = (
            get_container_env_var("PREPEND_PATH") + ":" + os.environ["PATH"]
        )

    def setup_toil_directories(self, workdir: str) -> tuple[str, str]:
        dir_coordination = os.path.join(workdir, "coordination")
        try:
            os.mkdir(dir_coordination)
        except FileExistsError:
            print("Coordination directory already exists, not overwriting.")

        dir_slurmlogs = os.path.join(get_container_env_var("LOGSDIR"), "slurmlogs")
        try:
            os.mkdir(dir_slurmlogs)
        except FileExistsError:
            print("Slurm log directory already exists, not overwriting.")

        return (dir_coordination, dir_slurmlogs)


logger = structlog.getLogger()

if "LINC_DATA_ROOT" not in os.environ:
    logger.critical(
        "LINC_DATA_ROOT environment variable has not been set. This is needed for pipeline execution and generating $LINC_DATA_ROOT/.versions file."
    )
    sys.exit(-1)
app = App(group="LOFAR")


@app.command()
def delay_calibration(
    mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    delay_calibrator: Annotated[
        dict,
        Parameter(
            help="A delay calibrator catalogue in CSV format.", converter=cwl_file
        ),
    ],
    image_catalogue: Annotated[
        Optional[dict],
        Parameter(
            help="The image catalogue (in FITS or CSV format) containing the target directions.",
            converter=cwl_file,
        ),
    ] = cwl_file(str, [Token(value="lotss_catalogue.csv")]),
    ATeam_skymodel: Annotated[
        Optional[dict],
        Parameter(help="File path to the A-Team skymodel.", converter=cwl_file),
    ] = cwl_file(
        str,
        [
            Token(
                value=os.path.join(
                    os.environ["LINC_DATA_ROOT"], "skymodels/A-Team.skymodel"
                )
            )
        ],
    ),
    rfi_strategy: Annotated[
        Optional[dict],
        Parameter(
            help="File path to the strategy file for AOFlagger.", converter=cwl_file
        ),
    ] = cwl_file(
        str,
        [
            Token(
                value=os.path.join(
                    os.environ["LINC_DATA_ROOT"], "rfistrategies/lofar-hba-wideband.lua"
                )
            )
        ],
    ),
    configfile: Annotated[
        Optional[dict],
        Parameter(
            help="Settings for the delay calibration in delay_solve.",
            converter=cwl_file,
        ),
    ] = cwl_file(
        str,
        [
            Token(
                value=os.path.join(
                    os.environ["VLBI_DATA_ROOT"], "pipeline_config_files", "facetselfcal_config.txt"
                )
            )
        ],
    ),
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".MS",
    solset: Annotated[
        Optional[dict],
        Parameter(
            help="The solution tables generated by the LINC target pipeline in an HDF5 format.",
            converter=cwl_file,
        ),
    ] = None,
    filter_baselines: Annotated[
        Optional[str],
        Parameter(help="The default filter constraints for the dp3_prep_target step."),
    ] = "*&",
    flag_baselines: Annotated[
        Optional[List[str]],
        Parameter(
            help="The baselines to be flagged by DP3. Can be a pattern, e.g. [ CS013HBA*&&* ]."
        ),
    ] = None,
    phasesol: Annotated[
        Optional[str],
        Parameter(
            help="The name of the target solution table to use from the solset input."
        ),
    ] = "TGSSphase",
    reference_stationSB: Annotated[
        Optional[int],
        Parameter(
            help="Subbands are concatenated in the concatenate-flag workflow relative to this station subband."
        ),
    ] = 104,
    number_cores: Annotated[
        Optional[int],
        Parameter(
            help="Number of cores to use per job for tasks with high I/O or memory."
        ),
    ] = 12,
    max_dp3_threads: Annotated[
        Optional[int], Parameter(help="The number of threads per DP3 process.")
    ] = 5,
    ddf_solsdir: Annotated[
        Optional[dict],
        Parameter(
            help="[Required if subtracting LoTSS] Path to the SOLSDIR directory of the DDF-pipeline run, where most of the calibration solutions are stored.",
            converter=cwl_dir,
        ),
    ] = None,
    ddf_rundir: Annotated[
        Optional[dict],
        Parameter(
            help="[Required if subtracting LoTSS] Path to the directory of the DDF-pipeline run where files required for the subtract can be found.",
            converter=cwl_dir,
        ),
    ] = None,
    box_size: Annotated[
        Optional[float],
        Parameter(
            help="[Required if subtracting LoTSS] Box size, in degrees, outside of which to subtract the LoTSS model from the data."
        ),
    ] = 2.5,
    subtract_chunk_hours: Annotated[
        Optional[float],
        Parameter(
            help="The range of time to predict the LoTSS model for at once. Lowering this value reduces memory footprint at the (possible) cost of increased runtime and vice versa."
        ),
    ] = 0.5,
    do_subtraction: Annotated[
        Optional[bool],
        Parameter(
            help="When set to true, the LoTSS model will be subtracted from the DDF corrected data."
        ),
    ] = False,
    do_validation: Annotated[
        Optional[bool],
        Parameter(help="Validate DI calibration."),
    ] = True,
    apply_delay_solutions: Annotated[
        Optional[bool],
        Parameter(help="Appy the delay calibration solutions to the full MS."),
    ] = False,
    model_image: Annotated[
        Optional[dict],
        Parameter(
            help="Image to generate an initial delay calibration model from.",
            converter=cwl_file,
        ),
    ] = None,
    rm_correction: Annotated[
        Optional[Literal["spinifex", "RMextract"]],
        Parameter(
            help="Name of the rotation measure solution table.",
        ),
    ] = None,
    do_auto_delay_selection: Annotated[
        Optional[bool],
        Parameter(
            help="Automatically select the best candidate delay calibrator(s) based on phasediff scores."
        ),
    ] = False,
    select_best_n_delay_calibrators: Annotated[
        Optional[int],
        Parameter(help="Select these N best scoring delay calibrators."),
    ] = 1,
    starting_skymodel: Annotated[
        Optional[list[dict]],
        Parameter(
            help="Optional starting models in BBS-compatible text format for starting delay calibration.",
            converter=cwl_file,
        ),
    ] = None,
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    outdir: Annotated[
        str,
        Parameter(help="Directory to move outputs to."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "72:00:00",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
    record_toil_stats: Annotated[
        bool,
        Parameter(
            help="Use Toil's stats flag to record statistics. N.B. this disables cleanup of successful steps; make sure there is enough disk space until the end of the run."
        ),
    ] = False,
):
    args = locals()
    logger.info("Generating VLBI delay-calibration config")
    config = VLBIJSONConfig(
        args["mspath"],
        ms_suffix=args["ms_suffix"],
        prefac_h5parm=args["solset"],
        ddf_solsdir=args["ddf_solsdir"],
        outdir=outdir,
    )
    unneeded_keys = [
        "mspath",
        "ms_suffix",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
        "restart",
        "record_toil_stats",
    ]
    args_for_linc = args.copy()
    for key in unneeded_keys:
        args_for_linc.pop(key)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_delay-calibration.json")
    if args["record_toil_stats"] and args["runner"] != "toil":
        logger.critical("--record-toil-stats needs '--runner toil'.")
        sys.exit(-1)
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
            record_stats=args["record_toil_stats"],
        )


@app.command()
def dd_calibration(
    mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    source_catalogue: Annotated[
        dict,
        Parameter(
            help="The image catalogue (in FITS or CSV format) containing the target directions.",
            converter=cwl_file,
        ),
    ],
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".MS",
    delay_solset: Annotated[
        Optional[dict],
        Parameter(
            help="The solution tables generated by the VLBI delay calibration workflow in an HDF5 format.",
            converter=cwl_file,
        ),
    ] = None,
    dd_dutch_solutions: Annotated[
        Optional[dict],
        Parameter(
            help="Provide already obtained direction-dependent h5parm solutions for the Dutch LOFAR array to pre-apply before international LOFAR calibration.",
            converter=cwl_file,
        ),
    ] = None,
    max_dp3_threads: Annotated[
        Optional[int],
        Parameter(
            help="Number of cores to use per job for tasks with high I/O or memory."
        ),
    ] = 4,
    numbands: Annotated[
        Optional[int],
        Parameter(help="The number of bands to group. -1 means all bands."),
    ] = -1,
    truncateLastSBs: Annotated[
        Optional[bool],
        Parameter(
            help="Whether to truncate the last subbands of the MSs to the same length."
        ),
    ] = True,
    dd_selection: Annotated[
        Optional[bool],
        Parameter(
            help="If set to true the pipeline will perform direction-dependent calibrator selection."
        ),
    ] = True,
    phasediff_score: Annotated[
        float,
        Parameter(
            help="Phasediff-score for calibrator selection <2.3 good for DD-calibrators and <0.7 good for DI-calibrators. Only used when dd_selection==true."
        ),
    ] = 2.3,
    custom_phasediff_score_csv: Annotated[
        Optional[dict],
        Parameter(
            help="CSV-formatted file with phasediff_scores. Overwrites the scores one generated in the DD-selection.",
            converter=cwl_file,
        ),
    ] = None,
    peak_flux_cut: Annotated[
        float,
        Parameter(help="Peak flux (Jy/beam) cut to pre-select sources from catalogue."),
    ] = 0.025,
    model_cache: Annotated[
        Optional[str],
        Parameter(help="Neural network cache directory."),
    ] = None,
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    outdir: Annotated[
        str,
        Parameter(help="Directory to move outputs to."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
    record_toil_stats: Annotated[
        bool,
        Parameter(
            help="Use Toil's stats flag to record statistics. N.B. this disables cleanup of successful steps; make sure there is enough disk space until the end of the run."
        ),
    ] = False,
):
    args = locals()
    cat = Table.read(source_catalogue["path"])
    cat_modified = False
    for source in cat:
        try:
            parsed_input = re.findall(
                r"ILTJ\d{6}\.\d{2}[+\-]\d{6}\.\d{1}", source["Source_Name"]
            )[0]
        except IndexError:
            newname = ra_dec_to_iltj(source["RA"], source["DEC"])
            logger.info(
                f"Source {source['Source_Name']} does not adhere to ILTJhhmmss.ss+ddmmss.s convention. It will be renamed to {newname} to avoid matching problems."
            )
            source["Source_Name"] = newname
            cat_modified = True
    if cat_modified:
        shutil.copy(source_catalogue["path"], source_catalogue["path"] + ".bkp")
        cat.write(source_catalogue, overwrite=True)

    logger.info("Generating VLBI dd-calibration config")
    config = VLBIJSONConfig(args["mspath"], ms_suffix=args["ms_suffix"], outdir=outdir)
    unneeded_keys = [
        "mspath",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
        "record_toil_stats",
    ]
    args_for_linc = args.copy()
    for key in unneeded_keys:
        args_for_linc.pop(key)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_dd-calibration.json")
    if args["record_toil_stats"] and args["runner"] != "toil":
        logger.critical("--record-toil-stats needs '--runner toil'.")
        sys.exit(-1)
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
            record_stats=args["record_toil_stats"],
        )


@app.command()
def split_directions(
    mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".MS",
    image_cat: Annotated[
        dict,
        Parameter(
            help="The image catalogue (in FITS or CSV format) containing the target directions.",
            converter=cwl_file,
        ),
    ] = "lotss_catalogue.csv",
    configfile: Annotated[
        Optional[dict],
        Parameter(
            help="The configuration file to be used to run facetselfcal.py during the target_selfcal step.",
            converter=cwl_file,
        ),
    ] = None,
    delay_solset: Annotated[
        Optional[dict],
        Parameter(
            help="The solution tables generated by the VLBI delay calibration workflow in an HDF5 format.",
            converter=cwl_file,
        ),
    ] = None,
    max_dp3_threads: Annotated[
        Optional[int],
        Parameter(
            help="Number of cores to use per job for tasks with high I/O or memory."
        ),
    ] = 4,
    numbands: Annotated[
        Optional[int],
        Parameter(help="The number of bands to group. -1 means all bands."),
    ] = -1,
    truncateLastSBs: Annotated[
        Optional[bool],
        Parameter(
            help="Whether to truncate the last subbands of the MSs to the same length."
        ),
    ] = True,
    do_selfcal: Annotated[
        Optional[bool],
        Parameter(help="Whether to do selfcal on the direction concat MSs."),
    ] = False,
    dd_selection: Annotated[
        Optional[bool],
        Parameter(
            help="If set to true the pipeline will perform direction-dependent calibrator selection."
        ),
    ] = False,
    phasediff_score: Annotated[
        float,
        Parameter(
            help="Phasediff-score for calibrator selection <2.3 good for DD-calibrators and <0.7 good for DI-calibrators. Only used when dd_selection==true."
        ),
    ] = 2.3,
    peak_flux_cut: Annotated[
        float,
        Parameter(
            help="Peak flux (Jy/beam) cut to pre-select sources from catalogue. Default at 0.0 is no peak flux selection."
        ),
    ] = 0.0,
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
):
    args = locals()
    logger.info("Generating VLBI split-directions config")
    config = VLBIJSONConfig(
        args["mspath"],
        ms_suffix=args["ms_suffix"],
    )
    unneeded_keys = [
        "mspath",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
    ]
    args_for_linc = args.copy()
    for key in unneeded_keys:
        args_for_linc.pop(key)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_split-directions.json")
    if args["record_toil_stats"] and args["runner"] != "toil":
        logger.critical("--record-toil-stats needs '--runner toil'.")
        sys.exit(-1)
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
        )

@app.command()
def polarization_imaging(
	mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".ms",
    pixel_scale: Annotated[
        Optional[str],
        Parameter(help="Pixel sampling for imaging in WSClean"),
    ] = "0.075arcsec",
    taper: Annotated[
        Optional[str],
        Parameter(help="Gaussian taper for shaping the PSF in WSClean"),
    ] = "0.3arcsec",
    image_size: Annotated[
        Tuple[int,int],
        Parameter(help="Image size in number of pixels [x,y]"),
    ] = (2000,2000),
    num_channels: Annotated[
        Optional[int],
        Parameter(help="The number of channels to image for each Stokes"),
    ] = 480,
    stokes: Annotated[
        Optional[str],
        Parameter(help="Stokes to image. Any combination of I, Q, U, and V"),
    ] = "IQUV",
    rmtools_max_lam2: Annotated[
        Optional[float],
        Parameter(help="Maximum lambda-squared value for rmsynth3d (-l)."),
    ] = 150,
    rmtools_dlam2: Annotated[
        Optional[float],
        Parameter(help="Lambda-squared channel width for rmsynth3d (-d)."),
    ] = 0.3,
    rmtools_output_prefix: Annotated[
        Optional[str],
        Parameter(help="Prefix for RM-Tools output products. Defaults to Stokes Q basename."),
    ] = "target",
    rmtools_extra_args: Annotated[
        Optional[str],
        Parameter(help="Extra arguments passed to rmsynth3d."),
    ] = None,
    outdir: Annotated[
        str,
        Parameter(help="Directory to move outputs to."),
    ] = os.getcwd(),
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
):
    args = locals()
    logger.info("Generating VLBI polarization-imaging config")

    config = VLBIJSONConfig(
        args["mspath"],
        ms_suffix=args["ms_suffix"], outdir=outdir
    )
    unneeded_keys = [
        "mspath",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
        "restart",
        "outdir",
    ]
    args_for_linc = args.copy()

    for key in unneeded_keys:
        args_for_linc.pop(key,None)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_polarization-imaging.json")
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
        )


@app.command()
def setup(
    mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    solset: Annotated[
        dict,
        Parameter(
            help="The solution tables generated by the LINC target pipeline in an HDF5 format.",
            converter=cwl_file,
        ),
    ],
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".MS",
    filter_baselines: Annotated[
        Optional[str],
        Parameter(help="The default filter constraints for the dp3_prep_target step."),
    ] = "*&",
    flag_baselines: Annotated[
        Optional[List[str]],
        Parameter(
            help="The baselines to be flagged by DP3. Can be a pattern, e.g. [ CS013HBA*&&* ]."
        ),
    ] = None,
    phasesol: Annotated[
        Optional[str],
        Parameter(
            help="The name of the target solution table to use from the solset input."
        ),
    ] = "TGSSphase",
    min_separation: Annotated[
        Optional[int],
        Parameter(
            help="The minimal accepted angular distance to an A-team source on the sky in degrees."
        ),
    ] = 30,
    number_cores: Annotated[
        Optional[int],
        Parameter(
            help="The minimum number of cores that should be available for steps that require high I/O."
        ),
    ] = 12,
    max_dp3_threads: Annotated[
        Optional[int],
        Parameter(help="The maximum number of threads DP3 should use per process."),
    ] = 5,
    clip_sources: Annotated[
        Optional[List[str]],
        Parameter(
            help="The patches of sources that should be flagged. These should be present in the LINC skymodel."
        ),
    ] = ["VirA_Gaussian", "CygA_Gaussian", "CasA_Gaussian", "TauA_Gaussian"],
    ATeam_skymodel: Annotated[
        Optional[dict],
        Parameter(
            help="File path to the A-Team skymodel.",
            converter=cwl_file,
        ),
    ] = cwl_file(
        str,
        [
            Token(
                value=os.path.join(
                    os.environ["LINC_DATA_ROOT"], "skymodels", "A-Team.skymodel"
                )
            )
        ],
    ),
    rm_correction: Annotated[
        Optional[Literal["spinifex", "RMextract"]],
        Parameter(
            help="Name of the rotation measure solution table.",
        ),
    ] = None,
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
):
    args = locals()
    logger.info("Generating VLBI setup config")
    config = VLBIJSONConfig(
        args["mspath"],
        ms_suffix=args["ms_suffix"],
        prefac_h5parm=args["solset"],
    )
    unneeded_keys = [
        "mspath",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
        "restart",
    ]
    args_for_linc = args.copy()
    for key in unneeded_keys:
        args_for_linc.pop(key)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_setup.json")
    if args["record_toil_stats"] and args["runner"] != "toil":
        logger.critical("--record-toil-stats needs '--runner toil'.")
        sys.exit(-1)
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
        )


@app.command()
def concatenate_flag(
    mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".MS",
    numbands: Annotated[
        Optional[int],
        Parameter(
            help="The number of files that have to be grouped together in frequency."
        ),
    ] = 10,
    firstSB: Annotated[
        Optional[int],
        Parameter(
            help="If set, reference the grouping of files to this station subband."
        ),
    ] = None,
    max_dp3_threads: Annotated[
        Optional[int],
        Parameter(
            help="The maximum number of threads that DP3 should use per process."
        ),
    ] = 5,
    aoflagger_memory_fraction: Annotated[
        Optional[int],
        Parameter(
            help="The fraction of the node's memory that will be used by AOFlagger (and should be available before an AOFlagger job can start)."
        ),
    ] = 15,
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
):
    args = locals()
    logger.info("Generating VLBI concatenate-flag config")
    config = VLBIJSONConfig(
        args["mspath"],
        ms_suffix=args["ms_suffix"],
    )
    unneeded_keys = [
        "mspath",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
    ]
    args_for_linc = args.copy()
    for key in unneeded_keys:
        args_for_linc.pop(key)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_concatenate-flag.json")
    if args["record_toil_stats"] and args["runner"] != "toil":
        logger.critical("--record-toil-stats needs '--runner toil'.")
        sys.exit(-1)
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
        )


@app.command()
def phaseup_concat(
    mspath: Annotated[str, Parameter(help="Directory where MSes are located.")],
    delay_calibrator: Annotated[
        dict,
        Parameter(
            help="Catalogue file with information on in-field calibrator.",
            converter=cwl_file,
        ),
    ],
    configfile: Annotated[
        dict,
        Parameter(
            help="Settings for the delay calibration in delay_solve.",
            converter=cwl_file,
        ),
    ],
    ms_suffix: Annotated[
        str, Parameter(help="Extension to look for when searching `mspath` for MSes.")
    ] = ".MS",
    numbands: Annotated[
        Optional[int],
        Parameter(help="The number of files that have to be grouped together."),
    ] = -1,
    firstSB: Annotated[
        Optional[int],
        Parameter(
            help="If set, reference the grouping of files to this station subband."
        ),
    ] = None,
    max_dp3_threads: Annotated[
        Optional[int],
        Parameter(help="The maximum number of threads DP3 should use per process."),
    ] = 5,
    number_cores: Annotated[
        Optional[int],
        Parameter(
            help="Number of cores to use per job for tasks with high I/O or memory."
        ),
    ] = 12,
    config_only: Annotated[
        bool,
        Parameter(help="Only generate the config file, do not run it."),
    ] = False,
    scheduler: Annotated[
        str,
        Parameter(help="System scheduler to use."),
    ] = "singleMachine",
    runner: Annotated[
        str,
        Parameter(help="CWL runner to use."),
    ] = "cwltool",
    rundir: Annotated[
        str,
        Parameter(help="Directory to run in."),
    ] = os.getcwd(),
    slurm_queue: Annotated[
        str,
        Parameter(help="Slurm queue to run jobs on."),
    ] = "",
    slurm_account: Annotated[
        str,
        Parameter(help="Slurm account to use."),
    ] = "",
    slurm_time: Annotated[
        str,
        Parameter(help="Slurm time limit to use."),
    ] = "",
    restart: Annotated[
        bool,
        Parameter(help="Restart a toil workflow."),
    ] = False,
):
    args = locals()
    logger.info("Generating VLBI phaseup-concat config")
    config = VLBIJSONConfig(
        args["mspath"],
        ms_suffix=args["ms_suffix"],
    )
    unneeded_keys = [
        "mspath",
        "config_only",
        "scheduler",
        "runner",
        "rundir",
        "slurm_queue",
        "slurm_account",
        "slurm_time",
    ]
    args_for_linc = args.copy()
    for key in unneeded_keys:
        args_for_linc.pop(key)
    for key, val in args_for_linc.items():
        config.add_entry(key, val)
    config.save("mslist_VLBI_phaseup-concat.json")
    if args["record_toil_stats"] and args["runner"] != "toil":
        logger.critical("--record-toil-stats needs '--runner toil'.")
        sys.exit(-1)
    if not args["config_only"]:
        config.run_workflow(
            runner=args["runner"],
            scheduler=args["scheduler"],
            slurm_params={
                "queue": args["slurm_queue"],
                "account": args["slurm_account"],
                "time": args["slurm_time"],
            },
            workdir=args["rundir"],
            restart=args["restart"],
        )


if __name__ == "__main__":
    print("Hello")
    app()
