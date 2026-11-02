
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# ==========================================
# LOAD PRETRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True
    )
    return model


# ==========================================
# ANIMAL CATEGORY MAPPING
# ==========================================

# Common dog breeds and dog-related labels
DOG_LABELS = {
    "german shepherd",
    "labrador retriever",
    "golden retriever",
    "beagle",
    "pug",
    "rottweiler",
    "doberman",
    "boxer",
    "bulldog",
    "poodle",
    "chihuahua",
    "siberian husky",
    "great dane",
    "border collie",
    "dalmatian",
    "pomeranian",
    "samoyed",
    "shih-tzu",
    "corgi",
    "terrier",
    "spaniel",
    "retriever",
    "shepherd",
    "hound",
    "mastiff",
    "pinscher",
    "schnauzer",
    "setter",
    "pointer",
    "malamute",
    "collie",
    "whippet",
    "greyhound",
    "dingo",
    "affenpinscher",
    "airedale",
    "basenji",
    "basset",
    "bloodhound",
    "borzoi",
    "briard",
    "bouvier",
    "cairn",
    "chow",
    "clumber",
    "eskimo dog",
    "keeshond",
    "kelpie",
    "komondor",
    "kuvasz",
    "leonberg",
    "newfoundland",
    "papillon",
    "pekinese",
    "pembroke",
    "schipperke",
    "shetland sheepdog",
    "shiba inu",
    "vizsla",
    "weimaraner",
    "wolfhound",
    "yorkshire terrier",
    "toy terrier",
    "french bulldog",
    "boston bull",
    "bernese mountain dog",
    "old english sheepdog",
    "afghan hound",
    "scottish deerhound",
    "irish wolfhound",
    "norwegian elkhound",
    "english setter",
    "irish setter",
    "cocker spaniel",
    "springer spaniel",
    "flat-coated retriever",
    "curly-coated retriever",
    "chesapeake bay retriever",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "american staffordshire terrier",
    "staffordshire bullterrier",
    "west highland white terrier",
    "wire-haired fox terrier",
    "soft-coated wheaten terrier",
    "tibetan mastiff",
    "bull mastiff",
    "american bulldog",
    "australian terrier",
    "bedlington terrier",
    "border terrier",
    "lakeland terrier",
    "sealyham terrier",
    "silky terrier",
    "scotch terrier",
    "irish terrier",
    "entlebucher",
    "appenzeller",
    "greater swiss mountain dog",
    "mexican hairless",
    "african hunting dog",
    "dhole",
    "redbone",
    "bluetick",
    "black-and-tan coonhound",
    "walker hound",
    "italian greyhound",
    "ibizan hound",
    "saluki",
    "brittany spaniel",
    "sussex spaniel",
    "gordon setter",
    "welsh springer spaniel",
    "brabancon griffon",
    "american eskimo dog",
}

# Common cat labels
CAT_LABELS = {
    "tabby",
    "tiger cat",
    "persian cat",
    "siamese cat",
    "egyptian cat",
    "lynx",
    "cougar",
    "snow leopard",
    "jaguar",
    "leopard",
    "lion",
    "tiger",
    "cheetah",
    "house cat",
    "wildcat",
}


def identify_animal(predicted_label):
    """
    Infer the animal category from the model's predicted label.
    This is a label-based estimate, not an independent animal detector.
    """

    name = predicted_label.lower().replace("_", " ").strip()

    # Check dog breeds first
    for dog_label in DOG_LABELS:
        if name == dog_label or name.endswith(" " + dog_label):
            return "Dog", predicted_label

    # Check cat labels
    for cat_label in CAT_LABELS:
        if name == cat_label or name.endswith(" " + cat_label):
            return "Cat", predicted_label

    # Check common animal words
    if "dog" in name or "puppy" in name:
        return "Dog", "Specific breed not identified"

    if "cat" in name or "kitten" in name:
        return "Cat", "Specific breed not identified"

    # Labels for other animals
    other_animals = {
        "bird": "Bird",
        "fish": "Fish",
        "horse": "Horse",
        "zebra": "Zebra",
        "elephant": "Elephant",
        "bear": "Bear",
        "rabbit": "Rabbit",
        "hamster": "Hamster",
        "fox": "Fox",
        "wolf": "Wolf",
        "monkey": "Monkey",
        "gorilla": "Gorilla",
        "deer": "Deer",
        "cow": "Cow",
        "sheep": "Sheep",
        "goat": "Goat",
        "pig": "Pig",
        "snake": "Snake",
        "turtle": "Turtle",
        "lizard": "Lizard",
        "frog": "Frog",
        "chicken": "Chicken",
        "duck": "Duck",
        "owl": "Owl",
        "eagle": "Eagle",
        "penguin": "Penguin",
    }

    for keyword, animal_type in other_animals.items():
        if keyword in name:
            return animal_type, "Specific breed not identified"

    return "Not identified", "Not applicable"


# ==========================================
# APPLICATION HEADER
# ==========================================

st.title("🖼️ AI Image Classifier")

st.write(
    "Upload an image and let artificial intelligence "
    "identify the object or animal."
)

st.info(
    "This application uses MobileNetV2, a pretrained "
    "image classification model trained on ImageNet."
)

# ==========================================
# LOAD MODEL
# ==========================================

try:
    with st.spinner("Loading the pretrained AI model..."):
        model = load_model()

except Exception as error:
    st.error("Unable to load the AI model.")
    st.write(
        "Check your internet connection and TensorFlow installation."
    )
    st.exception(error)
    st.stop()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔍 Classify Image", type="primary"):

            with st.spinner("Analyzing your image..."):

                # Resize image to the model's expected dimensions
                resized_image = image.resize((224, 224))

                # Convert image to a numerical array
                image_array = np.array(resized_image)

                # Add batch dimension
                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Apply MobileNetV2 preprocessing
                image_array = (
                    tf.keras.applications.mobilenet_v2
                    .preprocess_input(image_array)
                )

                # Run prediction
                predictions = model.predict(
                    image_array,
                    verbose=0
                )

                # Decode the top three ImageNet predictions
                results = (
                    tf.keras.applications.mobilenet_v2
                    .decode_predictions(predictions, top=3)[0]
                )

            # ==========================================
            # DISPLAY RESULTS
            # ==========================================

            label, description, confidence = results[0]

            predicted_name = description.replace(
                "_", " "
            ).title()

            animal_type, breed = identify_animal(
                predicted_name
            )

            st.success("Image Classification Complete!")

            st.subheader("🔍 Classification Results")

            st.write(
                f"**Predicted Label:** {predicted_name}"
            )

            st.write(
                f"**Animal Type:** {animal_type}"
            )

            st.write(
                f"**Breed / Specific Category:** {breed}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            # Display alternative predictions
            st.write("### Other Possible Predictions")

            for _, category, score in results:

                readable_category = category.replace(
                    "_", " "
                ).title()

                st.write(
                    f"**{readable_category}** "
                    f"— {score * 100:.2f}%"
                )

            # Explain the animal category limitation
            st.caption(
                "Animal type and breed are inferred from the "
                "model's predicted label. The model may make "
                "mistakes or may not recognize every animal "
                "or breed."
            )

    except Exception as error:
        st.error("Something went wrong while processing the image.")
        st.exception(error)


# ==========================================
# MODEL EXPLANATION
# ==========================================

with st.expander("How does this AI model work?"):

    st.write(
        """
        MobileNetV2 is a pretrained image classification model.

        1. The uploaded image is resized to 224 × 224 pixels.
        2. The image is converted into numerical data.
        3. The model analyzes visual patterns such as shapes,
           textures, and other image features.
        4. It calculates prediction scores for categories
           learned during its training.
        5. The application displays the top predicted labels.

        The model was pretrained on ImageNet, which contains
        many object and animal categories.

        Important: The model does not independently verify
        an animal's species or breed. Animal type and breed
        are inferred from the predicted category, and some
        predictions may be incorrect.
        """
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CodeOrbit Tech Internship Project | "
    "Image Classification Using a Pretrained Model"
)
