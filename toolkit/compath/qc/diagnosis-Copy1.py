


    def convert_to_geojson_mask(self, model_type):
        if self.metadata[model_type]["results_path"]["geojson"].exists():
            print(f"geojson mask for {model_type} already exists.")
            return
        if not self.metadata[model_type]["results_path"]["h5"].exists():
            print(
                f"h5 file for {model_type} does not exists, please run inference first"
            )
            return

        wkt_dict = h5.load_wkt_dict(self.metadata[model_type]["results_path"]["h5"])
        colors = get_rgb_colors(len(wkt_dict), cmap=get_cmap(6))
        geojson_features = []
        for idx, dict_item in enumerate(wkt_dict.items()):
            key, value = dict_item
            geojson_feature = geom_to_geojson(loads(value))
            geojson_feature["properties"] = {
                "objectType": "annotation",
                "name": key,
                "color": colors[idx],
            }
            geojson_features.append(geojson_feature)
        geojson_feature_collection = geojson.FeatureCollection(geojson_features)
        save_geojson(
            geojson_feature_collection,
            self.metadata[model_type]["results_path"]["geojson"],
        )

    def run_model_inference(
        self,
        model_type="tissue",
        patch_size=1024,
        batch_size=8,
        save_geojson_mask=True,
        show_progress=True,
        **args,
    ):
        if model_type == "tissue":
            self.dataloader_type = "all_coordinates"
            
        else:
            if model_type == 'pen':
                patch_size = 256
                
            self.dataloader_type = ""tissue_contact_coordinates""
            if self.tissue_geom is None:
                if self.metadata["tissue"]["results_path"]["h5"].exists():
                    tissue_wkt = h5.load_wkt_dict(
                        self.metadata["tissue"]["results_path"]["h5"]
                    )["combined"]
                    tissue_geom = loads(tissue_wkt)
                    self.set_tissue_geom(tissue_geom)
                else:
                    raise ValueError("tissue.h5 does not exist, run tissue model first.")

        if self.metadata[model_type]["results_path"]["h5"].exists():
            print(f"h5 data for {model_type} already exists.")
            return

        self.load_model(model_type)

        overlap_size = int(patch_size * (0.0625))
        context_size = 2 * overlap_size

        self.set_params(
            self.metadata[model_type]["mpp"],
            patch_size,
            overlap_size,
            context_size,
            slice_key=model_type,
        )
        self.set_slicer(slice_key=model_type)
        
        self.sph[model_type]["pred_dicts"] = self._inference_logic(
            model_type=model_type,
            show_progress=show_progress,
            batch_size=batch_size,
             **args,
        )

        processed_pred_dict = self._post_process_pred_dicts(model_type)

        wkt_dict = {}
        for key, value in processed_pred_dict.items():
            wkt_dict[key] = MultiPolygon(processed_pred_dict[key]).buffer(0).wkt

        h5.save_wkt_dict(wkt_dict, self.metadata[model_type]["results_path"]["h5"])

        if save_geojson_mask:
            self.convert_to_geojson_mask(model_type)

    def run_model_sequence(self, patch_size=1024, model_run_sequence=None, num_workers=4):
        if model_run_sequence is None:
            model_run_sequence = list(self.metadata.keys())

        if "tissue" in model_run_sequence:
            model_run_sequence.remove("tissue")
            model_run_sequence.insert(0, "tissue")
        else:
            if self.metadata["tissue"]['results_path']["h5"].exists():
                pass
            else:
                raise ValueError("No tissue model in the sequence and no tissue mask exists.")
        
        for model_type in model_run_sequence:
            self.run_model_inference(
                model_type, patch_size=patch_size, num_workers=num_workers
            )
    
        self._compile_model_results(model_run_sequence=model_run_sequence)

    def load_model(self, model_type):
        if model_type == "all":
            for key, data in self.metadata.items():
                if "model" not in data or data["model"] is None:
                    self._load_and_store_model(key, data["model_config"])
        else:
            if model_type in self.metadata:
                if (
                    "model" not in self.metadata[model_type]
                    or self.metadata[model_type]["model"] is None
                ):
                    self._load_and_store_model(
                        model_type, self.metadata[model_type]["model_config"]
                    )
                else:
                    print(f"Model '{model_type}' is already loaded.")
            else:
                raise KeyError(f"Model '{model_type}' not found in metadata.")
                
    def _compile_model_results(self, model_run_sequence=None):
        if model_run_sequence is None:
            model_run_sequence = list(self.metadata.keys())
        
        if "tissue" in model_run_sequence:
            model_run_sequence.remove("tissue")
            model_run_sequence.insert(0, "tissue")
        else:
            if self.metadata["tissue"]['results_path']["h5"].exists():
                pass
            else:
                raise ValueError("No tissue model in the sequence and no tissue mask exists.")
                    
        diagnose_statistics = []
        name_map = {}
        
        for model_type in model_run_sequence:
            name_map[model_type] = model_type.capitalize()
            
        for model_type in model_run_sequence:
            temp_dict = {}
            wkt_dict = h5.load_wkt_dict(self.metadata[model_type]["results_path"]["h5"])
            if "combined" in wkt_dict.keys(): 
                mgeom = loads(wkt_dict["combined"])
                temp_dict["model_type"] = name_map[model_type]
                temp_dict["area"] = round(mgeom.area*(self.wsi.mpp**2),2 )
            else:
                temp_dict["model_type"] = name_map[model_type]
                temp_dict["area"] = 0
        
            diagnose_statistics.append(temp_dict)
    
        diagnose_statistics = pd.DataFrame(diagnose_statistics)
    
        tissue_area = diagnose_statistics.loc[diagnose_statistics["model_type"] == name_map["tissue"], "area"].values[0]
        diagnose_statistics["% area"] = round(diagnose_statistics["area"] / tissue_area, 5)
        
        percent_useable_area = round(1 - diagnose_statistics.loc[diagnose_statistics["model_type"] != name_map["tissue"], "% area"].sum(), 2) * 100
    
        #plot
        labels = diagnose_statistics['model_type']
        sizes = diagnose_statistics['% area']
        plt.figure(figsize=(9, 7))
        wedges, _ = plt.pie(sizes, startangle=90, colors=plt.cm.Paired.colors)
        plt.title(f'Usable Area (%): {percent_useable_area}')
        plt.legend(wedges, labels, title="Regions", loc="center left", bbox_to_anchor=(0, 1))
        plt.axis('equal')
        plt.savefig(f"{self.qc_folder}/usable_area.jpg")
        plt.close()
    
        #save csv
        diagnose_statistics.to_csv(f"{self.qc_folder}/diagnose_statistics.csv", index = False)

    def _load_and_store_model(self, model_type, config):
        if config["architecture"] == "Unet++":
            model = smp.UnetPlusPlus(
                encoder_name=config["encoder_name"],
                encoder_weights=config["encoder_weights"],
                in_channels=config["in_channels"],
                classes=config["classes"],
            )

        else:
            raise ValueError(f"Architecture {config['architecture']} not implemented ")

        model.load_state_dict(
            torch.load(
                self.metadata[model_type]["path"],
                map_location=self.device,
                weights_only=True,
            )
        )
        model = model.eval().to(self.device)
        self.metadata[model_type]["model"] = model

    def _inference_logic(self, model_type, show_progress, batch_size, **args):

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            ):
                pred_dicts = []
                if self.dataloader_type == ""tissue_contact_coordinates"":
                    boundary = True
                else:
                    boundary = False
                    
                iterator = self.get_torch_dataloader(
                    self.dataloader_type, batch_size=batch_size, data_loading_mode=self.data_loading_mode, **args
                    )
                if show_progress:
                    iterator = tqdm(
                        iterator,
                        desc=f"Running {model_type} model",
                    )

                for batch in iterator:
                    if boundary:
                        batch, tissue_masks = batch

                    batch = batch.to(self.device) - 0.5
                    preds_batch = self.metadata[model_type]["model"](batch)
                    preds_batch = torch.argmax(preds_batch, 1)
                    if boundary:
                        tissue_masks = tissue_masks.to(self.device)
                        preds_batch *= tissue_masks
                        del tissue_masks

                    preds_batch = preds_batch.to(torch.uint8).cpu().numpy()

                    for pred_mask in preds_batch:
                        pred_dict = self._process_pred_mask(
                            pred_mask=pred_mask, model_type=model_type
                        )
                        pred_dicts.append(pred_dict)
                    del preds_batch, batch
        
        del iterator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_dicts

    def _process_pred_mask(self, pred_mask, model_type):
        shift_dims = self.sph[model_type]["params"]["shift_dims"]
        pred_dict = {}
        med_blur = medianBlur(pred_mask, 15)

        for key, value in self.metadata[model_type]["class_map"].items():
            if key == "bg":
                continue
            class_mask = np.uint8(med_blur == value).copy()
            contours, hierarchy = get_contours(
                class_mask[
                    shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]
                ]
            )
            pred_dict[key] = [contours, hierarchy]

        med_blur[med_blur != 0] = 1
        contours, hierarchy = get_contours(
            med_blur[shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]]
        )
        pred_dict["combined"] = [contours, hierarchy]

        return pred_dict

    def _post_process_pred_dicts(self, model_type):
        shift_dims = self.sph[model_type]["params"]["shift_dims"]
        scale_factor = self.sph[model_type]["params"]["factor1"]

        if self.dataloader_type == "all_coordinates":
            coordinates = self.sph[model_type]["all_coordinates"]
        else:
            coordinates = [coord for coord, _ in self.sph[model_type][self.dataloader_type]]
            
        pred_dicts = self.sph[model_type]["pred_dicts"]
        processed_pred_dict = {}

        assert len(coordinates) == len(
            pred_dicts
        ), "number of coordinates and predictions are unequal, can't merge back predictions together."

        for pred_dict, coordinate in zip(pred_dicts, coordinates):
            x, y = coordinate
            for key, value in pred_dict.items():
                contours, hierarchy = value
                if len(contours) == 0:
                    continue
                polys = get_shapely_poly(
                    contours,
                    hierarchy,
                    scale_factor=scale_factor,
                    shift_x=x + int(shift_dims[0] * scale_factor),
                    shift_y=y + int(shift_dims[1] * scale_factor),
                )
                if key in processed_pred_dict:
                    processed_pred_dict[key].extend(polys)
                else:
                    processed_pred_dict[key] = []
                    processed_pred_dict[key].extend(polys)

        return processed_pred_dict
