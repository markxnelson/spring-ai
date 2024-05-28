/*
 * Copyright 2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.springframework.ai.autoconfigure.oci.genai;

import org.springframework.ai.oci.OCIEmbeddingOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(OCIEmbeddingModelProperties.CONFIG_PREFIX)
public class OCIEmbeddingModelProperties {

	public static final String CONFIG_PREFIX = "spring.ai.oci.genai.embedding";

	private ServingMode servingMode;

	private String compartment;

	private String model;

	private boolean enabled;

	public OCIEmbeddingOptions getEmbeddingOptions() {
		return OCIEmbeddingOptions.builder()
			.withCompartment(compartment)
			.withModel(model)
			.withServingMode(servingMode.getMode())
			.build();
	}

	public ServingMode getServingMode() {
		return servingMode;
	}

	public void setServingMode(ServingMode servingMode) {
		this.servingMode = servingMode;
	}

	public String getCompartment() {
		return compartment;
	}

	public void setCompartment(String compartment) {
		this.compartment = compartment;
	}

	public String getModel() {
		return model;
	}

	public void setModel(String model) {
		this.model = model;
	}

	public boolean isEnabled() {
		return enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

}
