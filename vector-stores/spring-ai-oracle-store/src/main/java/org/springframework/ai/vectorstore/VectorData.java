/*
 * Copyright 2023 - 2024 the original author or authors.
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
package org.springframework.ai.vectorstore;

import oracle.sql.json.OracleJsonObject;

/**
 * Represents Vector data in Oracle.
 *
 * @author Corrado De Bari
 * @author Mark Nelson
 */
public class VectorData {

	private String id;

	private double[] embeddings;

	private String text;

	private OracleJsonObject metadata;

	private double distance;

	public String getText() {
		return text;
	}

	public void setText(String text) {
		this.text = text;
	}

	public OracleJsonObject getMetadata() {
		return metadata;
	}

	public void setMetadata(OracleJsonObject metadata) {
		this.metadata = metadata;
	}

	// Constructor
	public VectorData(String id, double[] embeddings, String text, OracleJsonObject metadata, double distance) {
		this.id = id;
		this.embeddings = embeddings;
		this.metadata = metadata;
		this.text = text;
		this.distance = distance;
	}

	// Getters and Setters
	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public double[] getEmbeddings() {
		return embeddings;
	}

	public void setEmbeddings(double[] embeddings) {
		this.embeddings = embeddings;
	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

}
