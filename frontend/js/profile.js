/* FitML — Profile setup page: measurement form + photo upload.
   Measurements are stored/sent in cm and kg (matching the trained model's
   expected units); unit toggles only change what's displayed/entered,
   converting back to cm/kg at submit time. */

const CM_PER_IN = 2.54;
const KG_PER_LB = 0.453592;

(function () {
  document.querySelectorAll(".accordion-header").forEach((header) => {
    header.addEventListener("click", () => {
      header.closest(".accordion-section").classList.toggle("open");
    });
  });

  const bandSelect = document.getElementById("bust_band");
  for (let b = 28; b <= 48; b += 2) {
    const opt = document.createElement("option");
    opt.value = String(b);
    opt.textContent = String(b);
    bandSelect.appendChild(opt);
  }

  const toggleBtns = document.querySelectorAll("#bust-toggle button");
  const bandCupBox = document.getElementById("bust-band-cup");
  const chestBox = document.getElementById("bust-chest-cm");
  let bustMode = "band_cup";

  toggleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      bustMode = btn.getAttribute("data-mode");
      toggleBtns.forEach((b) => b.classList.toggle("active", b === btn));
      bandCupBox.style.display = bustMode === "band_cup" ? "grid" : "none";
      chestBox.style.display = bustMode === "chest_cm" ? "block" : "none";
    });
  });

  // ---------------------------------------------------------- photo upload

  const PHOTO_SLOTS = ["front", "side", "back"];
  const selectedPhotos = { front: null, side: null, back: null };

  document.querySelectorAll("[data-photo-slot]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const slot = btn.dataset.photoSlot;
      const inputId = btn.dataset.action === "camera" ? `photo-camera-${slot}` : `photo-file-${slot}`;
      document.getElementById(inputId).click();
    });
  });

  PHOTO_SLOTS.forEach((slot) => {
    const preview = document.getElementById(`photo-preview-${slot}`);
    function handleChange(input) {
      const file = input.files[0];
      if (!file) return;
      selectedPhotos[slot] = file;
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
    }
    document.getElementById(`photo-file-${slot}`).addEventListener("change", (e) => handleChange(e.target));
    document.getElementById(`photo-camera-${slot}`).addEventListener("change", (e) => handleChange(e.target));
  });

  // ------------------------------------------------------------ unit toggles

  const UNIT_FIELDS = [
    { key: "height", inputId: "height_cm", kind: "length" },
    { key: "weight", inputId: "weight_kg", kind: "weight" },
    { key: "waist", inputId: "waist_cm", kind: "length" },
    { key: "hip", inputId: "hip_cm", kind: "length" },
    { key: "chest", inputId: "chest_cm", kind: "length" },
  ];
  const fieldUnits = {};

  function convert(kind, value, fromUnit, toUnit) {
    if (fromUnit === toUnit) return value;
    if (kind === "length") {
      return fromUnit === "cm" ? value / CM_PER_IN : value * CM_PER_IN;
    }
    return fromUnit === "kg" ? value / KG_PER_LB : value * KG_PER_LB;
  }

  UNIT_FIELDS.forEach(({ key, inputId, kind }) => {
    fieldUnits[key] = kind === "length" ? "cm" : "kg";
    const toggle = document.querySelector(`.unit-toggle[data-field="${key}"]`);
    const input = document.getElementById(inputId);
    if (!toggle || !input) return;
    toggle.querySelectorAll("button").forEach((btn) => {
      btn.addEventListener("click", () => {
        const newUnit = btn.dataset.unit;
        const oldUnit = fieldUnits[key];
        if (newUnit === oldUnit) return;
        const val = parseFloat(input.value);
        if (!isNaN(val)) {
          input.value = Math.round(convert(kind, val, oldUnit, newUnit) * 10) / 10;
        }
        fieldUnits[key] = newUnit;
        toggle.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
      });
    });
  });

  function toCanonical(key, kind) {
    const field = UNIT_FIELDS.find((f) => f.key === key);
    const input = document.getElementById(field.inputId);
    const val = parseFloat(input.value);
    if (input.value === "" || isNaN(val)) return "";
    return String(convert(kind, val, fieldUnits[key], kind === "length" ? "cm" : "kg"));
  }

  // --------------------------------------------------------------- submit

  const form = document.getElementById("profile-form");
  const submitBtn = document.getElementById("submit-btn");
  const errorEl = document.getElementById("form-error");
  const successEl = document.getElementById("form-success");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.style.display = "none";
    successEl.style.display = "none";

    const name = document.getElementById("shopper_name").value.trim();
    const heightCm = toCanonical("height", "length");
    if (!name) {
      errorEl.textContent = "Please enter your name.";
      errorEl.style.display = "block";
      return;
    }
    if (!selectedPhotos.front) {
      errorEl.textContent = "Please upload a front photo.";
      errorEl.style.display = "block";
      return;
    }
    if (!heightCm) {
      errorEl.textContent = "Height is required.";
      errorEl.style.display = "block";
      return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("photo", selectedPhotos.front);
    if (selectedPhotos.side) formData.append("photo_side", selectedPhotos.side);
    if (selectedPhotos.back) formData.append("photo_back", selectedPhotos.back);
    formData.append("height_cm", heightCm);

    const weightKg = toCanonical("weight", "weight");
    if (weightKg) formData.append("weight_kg", weightKg);

    if (bustMode === "band_cup") {
      const band = document.getElementById("bust_band").value;
      const cup = document.getElementById("bust_cup").value;
      if (band) formData.append("bust_band", band);
      if (cup) formData.append("bust_cup", cup);
    } else {
      const chestCm = toCanonical("chest", "length");
      if (chestCm) formData.append("chest_cm", chestCm);
    }

    const waistCm = toCanonical("waist", "length");
    if (waistCm) formData.append("waist_cm", waistCm);
    const hipCm = toCanonical("hip", "length");
    if (hipCm) formData.append("hip_cm", hipCm);
    const bodyType = document.getElementById("body_type").value;
    if (bodyType) formData.append("body_type", bodyType);

    formData.append("crop_face", document.getElementById("crop_face").checked ? "true" : "false");

    submitBtn.disabled = true;
    submitBtn.textContent = "Saving...";

    try {
      const res = await apiPostForm("/upload-profile", formData);
      setProfile({
        session_id: res.session_id,
        name: res.name,
        photo_coverage: res.photo_coverage,
        photo_url: res.photo_url,
        photo_side_url: res.photo_side_url || null,
        photo_back_url: res.photo_back_url || null,
        face_cropped: res.face_cropped,
        height_cm: heightCm,
        weight_kg: weightKg || null,
        waist_cm: waistCm || null,
        hip_cm: hipCm || null,
        body_type: bodyType || null,
      });
      successEl.textContent = "Profile saved. Taking you to the catalog...";
      successEl.style.display = "block";
      setTimeout(() => {
        window.location.href = "catalog.html";
      }, 700);
    } catch (err) {
      errorEl.textContent = err.message || "Something went wrong. Please try again.";
      errorEl.style.display = "block";
      submitBtn.disabled = false;
      submitBtn.textContent = "Save profile & continue";
    }
  });

  // Pre-fill from an existing profile, if the user is revisiting this page.
  const existing = getProfile();
  if (existing) {
    if (existing.name) document.getElementById("shopper_name").value = existing.name;
    if (existing.height_cm) document.getElementById("height_cm").value = existing.height_cm;
    if (existing.weight_kg) document.getElementById("weight_kg").value = existing.weight_kg;
    if (existing.waist_cm) document.getElementById("waist_cm").value = existing.waist_cm;
    if (existing.hip_cm) document.getElementById("hip_cm").value = existing.hip_cm;
    if (existing.body_type) document.getElementById("body_type").value = existing.body_type;
  }
})();
