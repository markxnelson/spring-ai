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
package org.springframework.ai.oci;

import java.util.ArrayList;
import java.util.List;

import com.oracle.bmc.generativeaiinference.model.DedicatedServingMode;
import com.oracle.bmc.generativeaiinference.model.EmbedTextDetails;
import com.oracle.bmc.generativeaiinference.model.OnDemandServingMode;
import com.oracle.bmc.generativeaiinference.model.ServingMode;
import com.oracle.bmc.generativeaiinference.requests.EmbedTextRequest;
import com.oracle.bmc.generativeaiinference.responses.EmbedTextResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.AbstractEmbeddingModel;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import com.oracle.bmc.generativeaiinference.GenerativeAiInferenceClient;
import org.springframework.ai.embedding.EmbeddingResponseMetadata;
import org.springframework.util.Assert;

public class OCIEmbeddingModel extends AbstractEmbeddingModel {

	private final GenerativeAiInferenceClient generativeAiClient;

	private final OCIEmbeddingOptions options;

	public OCIEmbeddingModel(GenerativeAiInferenceClient generativeAiClient, OCIEmbeddingOptions options) {
		Assert.notNull(generativeAiClient,
				"com.oracle.bmc.generativeaiinference.GenerativeAiInferenceClient must not be null");
		Assert.notNull(options, "Options must not be null");
		this.generativeAiClient = generativeAiClient;
		this.options = options;
	}

	@Override
	public EmbeddingResponse call(EmbeddingRequest request) {
		EmbedTextRequest embedTextRequest = generateEmbedTextRequest(request.getInstructions());
		return generateEmbeddingResponse(generativeAiClient.embedText(embedTextRequest));
	}

	@Override
	public List<Double> embed(Document document) {
		EmbedTextRequest embedTextRequest = generateEmbedTextRequest(List.of(document.getContent()));
		return toEmbeddings(generativeAiClient.embedText(embedTextRequest));
	}

	private ServingMode servingMode() {
		return switch (options.getServingMode()) {
			case "dedicated" -> DedicatedServingMode.builder().endpointId(options.getModel()).build();
			case "on-demand" -> OnDemandServingMode.builder().modelId(options.getModel()).build();
			default -> throw new IllegalArgumentException(
					"unknown serving mode for OCI embedding model: " + options.getServingMode());
		};
	}

	private EmbedTextRequest generateEmbedTextRequest(List<String> inputs) {
		EmbedTextDetails embedTextDetails = EmbedTextDetails.builder()
			.servingMode(servingMode())
			.compartmentId(options.getCompartment())
			.inputs(inputs)
			.truncate(EmbedTextDetails.Truncate.None)
			.build();
		return EmbedTextRequest.builder().embedTextDetails(embedTextDetails).build();
	}

	private EmbeddingResponse generateEmbeddingResponse(EmbedTextResponse embedTextResponse) {
		List<Embedding> embeddings = generateEmbeddingList(embedTextResponse);
		EmbeddingResponseMetadata metadata = generateMetadata(embedTextResponse);
		return new EmbeddingResponse(embeddings, metadata);
	}

	private EmbeddingResponseMetadata generateMetadata(EmbedTextResponse embedTextResponse) {
		EmbeddingResponseMetadata metadata = new EmbeddingResponseMetadata();
		metadata.put("model", embedTextResponse.getEmbedTextResult().getModelId());
		return metadata;
	}

	private List<Embedding> generateEmbeddingList(EmbedTextResponse embedTextResponse) {
		List<List<Float>> nativeData = embedTextResponse.getEmbedTextResult().getEmbeddings();
		List<Embedding> embeddings = new ArrayList<>();
		for (int i = 0; i < nativeData.size(); i++) {
			List<Double> data = toDoubleList(nativeData.get(i));
			embeddings.add(new Embedding(data, i));
		}
		return embeddings;
	}

	private List<Double> toEmbeddings(EmbedTextResponse embedTextResponse) {
		List<List<Float>> embeddings = embedTextResponse.getEmbedTextResult().getEmbeddings();
		if (embeddings.size() != 1) {
			throw new RuntimeException("expected exactly one OCI embedding result");
		}
		return toDoubleList(embeddings.get(0));
	}

	private List<Double> toDoubleList(List<Float> embeddings) {
		return embeddings.stream().map(Float::doubleValue).toList();
	}

}
