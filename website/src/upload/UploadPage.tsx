import React, { useState, useEffect } from 'react';

import RangeInput from '../common/RangeInput';
import LegendItem from '../common/LegendItem';

import { ReactCompareSlider, ReactCompareSliderImage, ReactCompareSliderHandle } from 'react-compare-slider';
import './UploadPage.css';

interface WeatherLabels {
    readonly snowy: number;
    readonly foggy: number;
    readonly rainy: number;
    readonly cloudy: number;
}

interface Name2NameMapping {
    [key: string] : string
}

interface ColorPallete {
    readonly [key: string]: number[];
}

const FORMATTED_WEATHER_CUES: Name2NameMapping = {
    background: "Background",
    black_clouds: "Black Clouds",
    white_clouds: "White Clouds",
    blue_sky: "Blue Sky",
    gray_sky: "Gray Sky",
    white_sky: "White Sky",
    fog: "Fog",
    sun: "Sun",
    snow: "Snow",
    shadow: "Shadow",
    wet_ground: "Wet Ground",
    shadow_snow: "Shadowed Snow",
};

function UploadPage() {
    const [loading, setLoading] = useState<boolean>(false)

    const [colorMap, setColorPallete] = useState<ColorPallete | undefined>(undefined);
    const [labels, setLabels] = useState<WeatherLabels | undefined>(undefined);
    const [file, setFile] = useState<File | undefined>(undefined);
    const [inputImageUrl, setInputImageUrl] = useState<string | undefined>(undefined);
    const [segmentationMaskUrl, setSegmentationMaskUrl] = useState<string | undefined>(undefined);

    useEffect(() => {
        if (labels) {
            console.log(labels)
        }
        if (colorMap) {
            console.log(colorMap)
        }
    }, [labels, colorMap])

    useEffect(() => {
        if (file) {
            handleFileUploaded(file)
        }
    }, [file])

    const handleFileUploadClick = (event: React.ChangeEvent<HTMLInputElement>) => {
        setLoading(true);
        const files = event.target.files;

        if (files) {
            const uploadedFile = files[0];
            setFile(uploadedFile);

            const reader = new FileReader();
            reader.onload = (e) => {
                const inputImageUrl = e.target?.result as string;
                setInputImageUrl(inputImageUrl);
            };

            reader.readAsDataURL(uploadedFile);
        }
    };

    const handleFileUploaded = (file: File) => {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/preprocess/predict', {
            method: 'POST',
            body: formData,
        }).then(response => {
            const labelsJson = response.headers.get('labels')!;
            const colorPalleteJson = response.headers.get('color_map')!;

            setLabels(JSON.parse(labelsJson))
            setColorPallete(JSON.parse(colorPalleteJson))

            console.log(JSON.parse(colorPalleteJson))

            return response.blob();
        }).then((data) => {
            const segmentationMaskUrl = URL.createObjectURL(data);
            setSegmentationMaskUrl(segmentationMaskUrl);
        }).catch(
            error => console.error(error)
        ).finally(() => {
            setLoading(false);
        });
    }

    return (
        <>
            <div className="image-button-box">
                <label htmlFor="image-upload" className="image-upload">
                    Upload Image
                </label>
                <input
                    id="image-upload"
                    type="file"
                    accept="image/jpg, image/jpeg"
                    onChange={handleFileUploadClick}
                />
            </div>
            <div className="image-controls">
                {loading && <div className="loader"></div>}

                {!loading &&
                    <div className="prediction-results">
                        <div className="label-results">
                            {labels &&
                                <div className="weather-label-container animate pop">
                                    <RangeInput
                                        id="snowy-range"
                                        color="white"
                                        min={0} max={1}
                                        step={0.25}
                                        defaultValue={labels.snowy}
                                    />
                                    <RangeInput
                                        id="cloudy-range"
                                        color="gray"
                                        min={0}
                                        max={1}
                                        step={0.25}
                                        defaultValue={labels.cloudy}
                                    />
                                    <RangeInput
                                        id="rainy-range"
                                        color="lightblue"
                                        min={0}
                                        max={1}
                                        step={0.5}
                                        defaultValue={labels.rainy}
                                    />
                                    <RangeInput
                                        id="foggy-range"
                                        color="red"
                                        min={0}
                                        max={1}
                                        step={0.5}
                                        defaultValue={labels.foggy}
                                    />

                                    <div className="weather-label">
                                        <div className="weather-label-name">cloudy</div>
                                        <div className="weather-label-value">{labels!.cloudy}</div>
                                    </div>
                                </div>
                            }
                        </div>
                        <div className="image-results">
                            {inputImageUrl && segmentationMaskUrl &&
                                <div className="image-slider-container animate pop">
                                    <ReactCompareSlider
                                        handle={
                                            <ReactCompareSliderHandle
                                                buttonStyle={{
                                                    border: 0,
                                                    backdropFilter: 'none',
                                                    WebkitBackdropFilter: 'none',
                                                    boxShadow: 'none',
                                                    color: 'black',
                                                    height: 0,
                                                    gap: 15,
                                                }}
                                                linesStyle={{ width: 8, color: 'black' }} />
                                        }

                                        itemOne={
                                            <ReactCompareSliderImage src={inputImageUrl} />
                                        }

                                        itemTwo={
                                            <ReactCompareSliderImage src={segmentationMaskUrl} />
                                        }
                                    />
                                </div>
                            }
                        </div>
                        {colorMap &&
                            <div>
                                <div className="weather-cue-legend">
                                    {Object.entries(colorMap).map(([name, color]) => {
                                        return <LegendItem name={FORMATTED_WEATHER_CUES[name]} color={`rgb(${color.join(',')})`} />
                                    })}
                                </div>
                            </div>
                        }
                    </div>
                }
            </div>
        </>
    )
}

export default UploadPage;
