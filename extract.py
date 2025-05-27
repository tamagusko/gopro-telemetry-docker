import os
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor


def extract_to_json(filepath, output_path):
    extractor = GoProTelemetryExtractor(filepath)
    extractor.open_source()
    extractor.extract_data_to_json(output_path, ["ACCL", "GYRO", "GPS9"])
    extractor.close_source()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    input_dir = "/data/input"
    output_dir = "/data/output"

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".mp4"):
            in_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(output_dir, base_name + ".json")
            extract_to_json(in_path, out_path)
